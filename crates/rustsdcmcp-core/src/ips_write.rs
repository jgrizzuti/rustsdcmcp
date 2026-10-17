//! Change-controlled writes to SDC IPS signatures and IPS profiles (#266).
//!
//! Deliberately a separate path from [`crate::SdcPreparedObjectWrite`] rather
//! than two more [`crate::WritableResource`] families:
//!
//! - **Scope.** Folding IPS into `prepare_sdc_object_write` would hand every
//!   existing object-write token IPS authoring without anyone re-minting it.
//!   Separate tools keep that a deliberate grant.
//! - **Guards the generic path cannot express.** A predefined signature (one
//!   imported from Juniper's signature bundle) must never be rewritten, and a
//!   write body must not smuggle server-assigned identity fields.
//! - **No delete yet.** Profile delete needs a "still referenced by a firewall
//!   rule" check, and #266 sequences deletes last.
//!
//! Like object writes, IPS writes have no SDC-side preview endpoint, so the
//! plan artifact is built locally: the exact request together with the
//! target's observed state beforehand, and apply refuses if that state has
//! since moved.
//!
//! An IPS write changes SDC's model only. Nothing reaches a device until a
//! policy referencing the profile is deployed through
//! `prepare_sdc_policy_deploy` → approve → `apply_sdc_change_set`.

use crate::{ResourceKind, SdcClient, SdcError, prepared::canonical_digest};
use async_trait::async_trait;
use mecmcp_audit::Attribution;
use mecmcp_changeset::{
    CommitOptions, CommitOutcome, DeviceTransaction, RollbackOutcome, RollbackRef, UnlockOutcome,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use tokio_util::sync::CancellationToken;

/// Hard cap on one serialized IPS-write envelope.
const MAX_ENVELOPE_BYTES: usize = 2 * 1024 * 1024;

/// Fields SDC assigns itself and a write body must therefore not carry.
///
/// `uuid` is allocated by SDC on create and addressed through the path on
/// update. `import_id` is the signature-bundle identifier the spec documents as
/// "applicable only for predefined signatures imported from signature
/// bundles"; letting a caller set it would let a custom signature masquerade as
/// predefined, or a predefined one be re-created under its bundle identity.
const SERVER_ASSIGNED_FIELDS: &[&str] = &["uuid", "import_id"];

/// IPS families this server may create and update.
///
/// Converts into [`ResourceKind`] one way only, for the same reason
/// [`crate::WritableResource`] does: exposing a family for reading must never
/// make it writable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IpsWriteFamily {
    /// A custom IPS signature, signature group, or anomaly
    /// (`/api/v1/ips_signatures`).
    Signature,
    /// An IPS profile (`/api/v1/ips_profiles`).
    Profile,
    /// An IPS rule inside one IPS profile
    /// (`/api/v1/ips_profiles/{profile_uuid}/ips_rules`). This is how
    /// signatures are added to a profile: a profile body carries only `name`
    /// and `description`.
    Rule,
}

impl IpsWriteFamily {
    /// Every writable IPS family, for exhaustive iteration in tests.
    pub const ALL: &'static [Self] = &[Self::Signature, Self::Profile, Self::Rule];

    /// The flat read family used for before-state and drift detection, or
    /// `None` for a rule, which is read through its parent profile.
    #[must_use]
    pub const fn resource_kind(self) -> Option<ResourceKind> {
        match self {
            Self::Signature => Some(ResourceKind::IpsSignatures),
            Self::Profile => Some(ResourceKind::IpsProfiles),
            Self::Rule => None,
        }
    }

    /// Whether this family lives under a parent IPS profile.
    #[must_use]
    pub const fn requires_profile(self) -> bool {
        matches!(self, Self::Rule)
    }

    /// Exact collection path segments. Flat families delegate to the read
    /// catalog, so a write and its drift-detection read can never address
    /// different paths; a rule's collection is the same one
    /// `list_sdc_ips_rules` reads. `None` when the parent profile is missing
    /// for a rule, or supplied for a flat family.
    #[must_use]
    pub fn collection_segments(self, profile_uuid: Option<&str>) -> Option<Vec<&str>> {
        match (self.resource_kind(), profile_uuid) {
            (Some(kind), None) => Some(kind.collection_segments().to_vec()),
            (None, Some(profile)) => Some(vec!["api", "v1", "ips_profiles", profile, "ips_rules"]),
            (Some(_), Some(_)) | (None, None) => None,
        }
    }
}

/// Which mutation one IPS write performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IpsWriteAction {
    /// Create a new signature or profile.
    Create,
    /// Replace an existing signature or profile by UUID.
    Update,
}

impl IpsWriteAction {
    /// Stable discriminator recorded in audit and change-control state.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Create => "ips_create",
            Self::Update => "ips_update",
        }
    }
}

/// Exact IPS write bound into a change-set action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdcPreparedIpsWrite {
    operation: String,
    action: IpsWriteAction,
    family: IpsWriteFamily,
    /// Parent profile of a rule. Absent (not `null`) for flat families, so a
    /// signature or profile envelope serializes exactly as before rules
    /// existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    profile_uuid: Option<String>,
    uuid: Option<String>,
    request: Value,
    before: Value,
    plan_digest: String,
}

impl SdcPreparedIpsWrite {
    /// Build a canonical, digest-bound signature or profile write.
    ///
    /// `before` is the target's observed state for update, and `Value::Null`
    /// for create. A rule needs its parent profile: use [`Self::new_rule`].
    ///
    /// # Errors
    ///
    /// Refuses shapes that do not match the action, bodies carrying
    /// server-assigned fields, updates to predefined signatures, invalid
    /// profile names, and oversized envelopes.
    pub fn new(
        action: IpsWriteAction,
        family: IpsWriteFamily,
        uuid: Option<String>,
        request: Value,
        before: Value,
    ) -> Result<Self, SdcError> {
        Self::build(action, family, None, uuid, request, before)
    }

    /// Build a canonical, digest-bound IPS rule write inside `profile_uuid`.
    ///
    /// # Errors
    ///
    /// Everything [`Self::new`] refuses, plus a rule body without at least one
    /// well-formed `ips_signature` reference or without a `then.action`.
    pub fn new_rule(
        action: IpsWriteAction,
        profile_uuid: String,
        uuid: Option<String>,
        request: Value,
        before: Value,
    ) -> Result<Self, SdcError> {
        Self::build(
            action,
            IpsWriteFamily::Rule,
            Some(profile_uuid),
            uuid,
            request,
            before,
        )
    }

    fn build(
        action: IpsWriteAction,
        family: IpsWriteFamily,
        profile_uuid: Option<String>,
        uuid: Option<String>,
        request: Value,
        before: Value,
    ) -> Result<Self, SdcError> {
        let plan = plan_artifact(
            action,
            family,
            profile_uuid.as_deref(),
            uuid.as_deref(),
            &request,
            &before,
        );
        let prepared = Self {
            operation: "ips_write".to_owned(),
            action,
            family,
            profile_uuid,
            uuid,
            request,
            before,
            plan_digest: canonical_digest(&plan)
                .map_err(|error| SdcError::PreparedChange(error.to_string()))?,
        };
        prepared.validate()?;
        Ok(prepared)
    }

    /// Parent profile of a rule write, absent for flat families.
    #[must_use]
    pub fn profile_uuid(&self) -> Option<&str> {
        self.profile_uuid.as_deref()
    }

    /// Product operation discriminator.
    #[must_use]
    pub fn operation(&self) -> &str {
        &self.operation
    }

    /// Which mutation this write performs.
    #[must_use]
    pub const fn action(&self) -> IpsWriteAction {
        self.action
    }

    /// Which IPS family this write targets.
    #[must_use]
    pub const fn family(&self) -> IpsWriteFamily {
        self.family
    }

    /// Target UUID, absent for create.
    #[must_use]
    pub fn uuid(&self) -> Option<&str> {
        self.uuid.as_deref()
    }

    /// Exact request body.
    #[must_use]
    pub const fn request(&self) -> &Value {
        &self.request
    }

    /// Target state observed at prepare time, `Null` for create.
    #[must_use]
    pub const fn before(&self) -> &Value {
        &self.before
    }

    /// Canonical SHA-256 digest binding this plan.
    #[must_use]
    pub fn plan_digest(&self) -> &str {
        &self.plan_digest
    }

    /// Human-readable plan showing what this write would change.
    #[must_use]
    pub fn plan(&self) -> Value {
        plan_artifact(
            self.action,
            self.family,
            self.profile_uuid.as_deref(),
            self.uuid.as_deref(),
            &self.request,
            &self.before,
        )
    }

    /// Revalidate shape, IPS guards, bounds, and digest integrity.
    ///
    /// Runs at prepare and again when the persisted envelope is staged for
    /// apply, so every guard here also holds against a tampered state file.
    ///
    /// # Errors
    ///
    /// Returns a stable error if the plan is not executable or was tampered.
    pub fn validate(&self) -> Result<(), SdcError> {
        if self.operation != "ips_write" {
            return Err(SdcError::PreparedChange(
                "prepared operation must be ips_write".to_owned(),
            ));
        }
        // Mirror the client's body rule so an unexecutable plan cannot be
        // approved; discovering it only at commit would burn a change set.
        let Some(fields) = self.request.as_object().filter(|fields| !fields.is_empty()) else {
            return Err(SdcError::PreparedChange(format!(
                "prepared {} requires a non-empty JSON object body",
                self.action.as_str()
            )));
        };
        let shape_ok = match self.action {
            IpsWriteAction::Create => self.uuid.is_none() && self.before.is_null(),
            IpsWriteAction::Update => self.uuid.is_some() && self.before.is_object(),
        };
        if !shape_ok {
            return Err(SdcError::PreparedChange(format!(
                "prepared {} does not match its required shape",
                self.action.as_str()
            )));
        }
        if let Some(uuid) = self.uuid.as_deref()
            && !is_allowed_identifier(uuid)
        {
            return Err(SdcError::PreparedChange(
                "prepared IPS UUID must be 1-256 characters from [A-Za-z0-9_-:+=]".to_owned(),
            ));
        }
        match (self.family.requires_profile(), self.profile_uuid.as_deref()) {
            (true, Some(profile)) if is_allowed_identifier(profile) => {}
            (true, _) => {
                return Err(SdcError::PreparedChange(
                    "an IPS rule write requires a profile_uuid of 1-256 characters from \
                     [A-Za-z0-9_-:+=]"
                        .to_owned(),
                ));
            }
            (false, Some(_)) => {
                return Err(SdcError::PreparedChange(
                    "profile_uuid applies only to IPS rule writes".to_owned(),
                ));
            }
            (false, None) => {}
        }
        if let Some(field) = SERVER_ASSIGNED_FIELDS
            .iter()
            .find(|field| fields.contains_key(**field))
        {
            return Err(SdcError::PreparedChange(format!(
                "IPS write body must not set the server-assigned field `{field}`"
            )));
        }
        // Both input schemas require `name`; requiring it on update too keeps a
        // PUT from silently blanking it, since SDC replaces the whole object.
        let Some(name) = fields.get("name").and_then(Value::as_str) else {
            return Err(SdcError::PreparedChange(
                "IPS write body requires a string `name`".to_owned(),
            ));
        };
        if self.family == IpsWriteFamily::Rule {
            validate_rule_body(fields)?;
        }
        if self.family == IpsWriteFamily::Profile && !is_valid_profile_name(name) {
            return Err(SdcError::PreparedChange(
                "IPS profile name must be 1-127 characters with no space, `#`, `&`, `;`, CR or LF"
                    .to_owned(),
            ));
        }
        if self.family == IpsWriteFamily::Signature {
            if let Some(kind) = fields.get("signature_type")
                && !kind.as_str().is_some_and(is_known_signature_type)
            {
                return Err(SdcError::PreparedChange(format!(
                    "IPS signature_type must be one of {}",
                    SIGNATURE_TYPES.join(", ")
                )));
            }
            if self.action == IpsWriteAction::Update {
                // Fail closed: an update needs positive evidence that the
                // target is custom, not merely the absence of a predefined mark.
                if !self
                    .before
                    .get("signature_type")
                    .and_then(Value::as_str)
                    .is_some_and(is_known_signature_type)
                {
                    return Err(SdcError::PreparedChange(
                        "refusing to update a signature whose observed signature_type is \
                         missing or unknown"
                            .to_owned(),
                    ));
                }
                if is_predefined_signature(&self.before) {
                    return Err(SdcError::PreparedChange(
                        "refusing to update a predefined signature imported from the \
                         signature bundle; only custom signatures are writable"
                            .to_owned(),
                    ));
                }
            }
        }
        let plan = self.plan();
        if canonical_digest(&plan).map_err(|error| SdcError::PreparedChange(error.to_string()))?
            != self.plan_digest
        {
            return Err(SdcError::PreparedChange(
                "prepared IPS write does not match its digest".to_owned(),
            ));
        }
        if serde_json::to_vec(self)
            .map_err(|_| SdcError::Serialization)?
            .len()
            > MAX_ENVELOPE_BYTES
        {
            return Err(SdcError::PreparedChange(
                "prepared IPS write exceeds the 2097152-byte limit".to_owned(),
            ));
        }
        Ok(())
    }
}

/// `signature_type` values the vendored spec allows.
const SIGNATURE_TYPES: &[&str] = &["anomaly", "signature", "chain", "static", "dynamic"];

fn is_known_signature_type(kind: &str) -> bool {
    SIGNATURE_TYPES.contains(&kind)
}

/// Bundle-provenance fields from the vendored spec's own field docs:
/// `import_id` "applicable only for predefined signatures imported from
/// signature bundles", and `activation_date`, set when a bundle activates a
/// signature. `signature_type` is the detection kind, shared by both origins.
const PREDEFINED_MARKERS: &[&str] = &["import_id", "activation_date"];

/// Whether an observed signature came from Juniper's signature bundle.
///
/// Any non-empty provenance marker counts, whatever its JSON type. Confirmed
/// on a lab tenant (2026-10-15): sampled bundle signatures of the anomaly,
/// chain and dynamic types all carried `import_id`, and signatures created
/// through this path carried neither marker.
fn is_predefined_signature(before: &Value) -> bool {
    PREDEFINED_MARKERS
        .iter()
        .any(|marker| match before.get(*marker) {
            None | Some(Value::Null) => false,
            Some(Value::String(text)) => !text.trim().is_empty(),
            Some(_) => true,
        })
}

/// Strict path-segment allowlist for IPS identifiers: 1-256 characters from
/// `[A-Za-z0-9_-:+=]`. A `/`, `.` or `%` can never reach a request path.
fn is_allowed_identifier(identifier: &str) -> bool {
    (1..=256).contains(&identifier.len())
        && identifier
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_-:+=".contains(&byte))
}

/// The UUID SDC allocated, read from a create response.
///
/// The spec documents both creates as returning the new object (`IpsSignature`
/// / `IpsProfile`, which carry `uuid`). A missing, empty, or non-string `uuid`
/// yields `None` rather than an error, because the write itself succeeded.
fn created_uuid_from(response: &Value) -> Option<String> {
    response
        .get("uuid")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|uuid| !uuid.is_empty())
        .map(str::to_owned)
}

/// The spec's `IpsProfileInput.name` format: `^[^#&;\r\n ]{1,127}$`.
fn is_valid_profile_name(name: &str) -> bool {
    let length = name.chars().count();
    (1..=127).contains(&length) && !name.contains(['#', '&', ';', '\r', '\n', ' '])
}

fn plan_artifact(
    action: IpsWriteAction,
    family: IpsWriteFamily,
    profile_uuid: Option<&str>,
    uuid: Option<&str>,
    request: &Value,
    before: &Value,
) -> Value {
    let mut plan = json!({
        "action": action,
        "family": family,
        "uuid": uuid,
        "before": before,
        "after": request,
    });
    // Bound into the digest for rules only, so the plan (and digest) of a
    // signature or profile write is byte-identical to before rules existed.
    if let (Some(profile), Value::Object(map)) = (profile_uuid, &mut plan) {
        map.insert("profile_uuid".to_owned(), Value::String(profile.to_owned()));
    }
    plan
}

/// Upper bound on signatures referenced by one rule.
const MAX_RULE_SIGNATURES: usize = 256;

/// A rule body must reference at least one signature by a well-formed UUID
/// and carry a `then.action`. `IPSRuleInput` requires `name`, `ips_signature`
/// and `then`; the spec leaves `then` fields unconstrained, so only `action`
/// is required here, as SDC's own rules always carry one.
fn validate_rule_body(fields: &serde_json::Map<String, Value>) -> Result<(), SdcError> {
    let refused = |reason: &str| Err(SdcError::PreparedChange(reason.to_owned()));
    let Some(Value::Array(references)) = fields.get("ips_signature") else {
        return refused("an IPS rule body requires an `ips_signature` array");
    };
    if references.is_empty() || references.len() > MAX_RULE_SIGNATURES {
        return refused("an IPS rule must reference 1-256 signatures in `ips_signature`");
    }
    for reference in references {
        if !reference
            .get("uuid")
            .and_then(Value::as_str)
            .is_some_and(is_allowed_identifier)
        {
            return refused(
                "every `ips_signature` entry needs a `uuid` of 1-256 characters from \
                 [A-Za-z0-9_-:+=]",
            );
        }
    }
    if !fields
        .get("then")
        .and_then(|then| then.get("action"))
        .and_then(Value::as_str)
        .is_some_and(|action| !action.trim().is_empty())
    {
        return refused("an IPS rule body requires `then.action`");
    }
    Ok(())
}

/// Signature UUIDs a rule body references, in order.
#[must_use]
pub fn rule_signature_uuids(request: &Value) -> Vec<String> {
    request
        .get("ips_signature")
        .and_then(Value::as_array)
        .map(|references| {
            references
                .iter()
                .filter_map(|reference| reference.get("uuid").and_then(Value::as_str))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Outcome of revalidating an IPS write immediately before commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpsValidationReport {
    /// Whether the envelope is intact and the target has not drifted.
    pub valid: bool,
    /// Which mutation was validated.
    pub action: IpsWriteAction,
    /// Which IPS family was validated.
    pub family: IpsWriteFamily,
    /// Whether the live target still matched its prepared `before` state.
    pub target_unchanged: bool,
}

/// Result of planning one IPS write.
#[derive(Debug, Clone, Serialize)]
pub struct IpsPrepareResult {
    /// Shared two-person change-set record.
    pub change_set: mecmcp_changeset::ChangeSetOutput,
    /// Exact IPS write bound by the plan digest.
    pub prepared_change: SdcPreparedIpsWrite,
}

/// Result of applying one approved IPS write.
#[derive(Debug, Clone, Serialize)]
pub struct IpsApplyResult {
    /// Shared operation identifier.
    pub operation_id: String,
    /// Before/after plan used as the management-plane diff.
    pub plan: Value,
    /// Drift and envelope validation result.
    pub validation: IpsValidationReport,
    /// Known, detached, or indeterminate commit disposition.
    pub outcome: CommitOutcome,
    /// UUID SDC allocated for a successful create, so a follow-up update can
    /// address it without searching the collection. `None` for update, and
    /// for a create whose response carried no UUID.
    pub created_uuid: Option<String>,
}

/// SDC implementation of the shared transaction contract for IPS writes.
#[derive(Clone)]
pub struct SdcIpsTransaction {
    client: SdcClient,
    expected_plan_digest: String,
    cancellation: CancellationToken,
    refused_before_write: Arc<AtomicBool>,
    created_uuid: Arc<OnceLock<String>>,
}

impl SdcIpsTransaction {
    /// Bind a transaction to one exact plan digest.
    #[must_use]
    pub fn new(
        client: SdcClient,
        expected_plan_digest: impl Into<String>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            client,
            expected_plan_digest: expected_plan_digest.into(),
            cancellation,
            refused_before_write: Arc::new(AtomicBool::new(false)),
            created_uuid: Arc::new(OnceLock::new()),
        }
    }

    /// Whether commit refused before issuing any request to SDC.
    ///
    /// Distinguishes "nothing was written, safe to discard" from "the write
    /// may have landed", which the coordinator's flattened error cannot.
    #[must_use]
    pub fn refused_before_write(&self) -> bool {
        self.refused_before_write.load(Ordering::SeqCst)
    }

    /// UUID SDC returned for a successful create, if any.
    ///
    /// The coordinator's [`CommitOutcome`] has no slot for it, so the
    /// transaction keeps it for the caller to report.
    #[must_use]
    pub fn created_uuid(&self) -> Option<String> {
        self.created_uuid.get().cloned()
    }

    /// Re-read the target and report whether it still matches its plan.
    ///
    /// Create has no prior object, so it is vacuously unchanged.
    async fn target_unchanged(&self, staged: &SdcPreparedIpsWrite) -> Result<bool, SdcError> {
        let (IpsWriteAction::Update, Some(uuid)) = (staged.action(), staged.uuid()) else {
            return Ok(true);
        };
        let current = self
            .client
            .get_ips_object(
                staged.family(),
                staged.profile_uuid(),
                uuid,
                &self.cancellation,
            )
            .await?;
        let digest = |value: &Value| {
            canonical_digest(value).map_err(|error| SdcError::PreparedChange(error.to_string()))
        };
        Ok(digest(&current)? == digest(staged.before())?)
    }
}

#[async_trait]
impl DeviceTransaction for SdcIpsTransaction {
    type Action = SdcPreparedIpsWrite;
    type Staged = SdcPreparedIpsWrite;
    type Diff = Value;
    type Validation = IpsValidationReport;
    type Error = SdcError;

    async fn fingerprint(&self) -> Result<String, Self::Error> {
        Ok(self.expected_plan_digest.clone())
    }

    /// Sole validation point for the persisted envelope.
    async fn stage(&self, actions: &[Self::Action]) -> Result<Self::Staged, Self::Error> {
        let [prepared] = actions else {
            return Err(SdcError::InvalidInput(
                "an SDC IPS write requires exactly one prepared change",
            ));
        };
        prepared.validate()?;
        if prepared.plan_digest() != self.expected_plan_digest {
            return Err(SdcError::PreparedChange(
                "prepared action does not match the approved plan digest".to_owned(),
            ));
        }
        Ok(prepared.clone())
    }

    async fn diff(&self, staged: &Self::Staged) -> Result<Self::Diff, Self::Error> {
        Ok(staged.plan())
    }

    /// Refuse the write if the target moved since it was prepared.
    async fn validate(&self, staged: &Self::Staged) -> Result<Self::Validation, Self::Error> {
        let target_unchanged = self.target_unchanged(staged).await?;
        if !target_unchanged {
            return Err(SdcError::TargetDrifted);
        }
        Ok(IpsValidationReport {
            valid: true,
            action: staged.action(),
            family: staged.family(),
            target_unchanged,
        })
    }

    async fn commit(
        &self,
        staged: &Self::Staged,
        _attribution: &Attribution,
        options: &CommitOptions,
    ) -> Result<CommitOutcome, Self::Error> {
        // Every exit until the request is issued is a pre-write refusal, so the
        // caller may discard instead of leaving a record that blocks the tenant.
        self.refused_before_write.store(true, Ordering::SeqCst);
        if options.confirm_timeout.is_some() {
            return Err(SdcError::InvalidInput(
                "SDC does not support confirmed IPS writes",
            ));
        }
        // Re-check drift immediately before writing: the coordinator releases
        // its guard between validate and commit, and that guard cannot exclude
        // a writer working directly against SDC. SDC has no conditional write,
        // so this narrows the window rather than closing it.
        if !self.target_unchanged(staged).await? {
            return Err(SdcError::TargetDrifted);
        }
        let family = staged.family();
        // From here the outcome is no longer knowably clean.
        self.refused_before_write.store(false, Ordering::SeqCst);
        let result = match (staged.action(), staged.uuid()) {
            (IpsWriteAction::Create, _) => {
                self.client
                    .create_ips_object(
                        family,
                        staged.profile_uuid(),
                        staged.request(),
                        &self.cancellation,
                    )
                    .await
            }
            (IpsWriteAction::Update, Some(uuid)) => {
                self.client
                    .update_ips_object(
                        family,
                        staged.profile_uuid(),
                        uuid,
                        staged.request(),
                        &self.cancellation,
                    )
                    .await
            }
            (IpsWriteAction::Update, None) => {
                self.refused_before_write.store(true, Ordering::SeqCst);
                return Err(SdcError::PreparedChange(
                    "prepared IPS update is missing its target UUID".to_owned(),
                ));
            }
        };
        match result {
            Ok(response) => {
                let created = (staged.action() == IpsWriteAction::Create)
                    .then(|| created_uuid_from(&response))
                    .flatten();
                let identity = match &created {
                    Some(uuid) => {
                        // Set once per transaction; a second commit cannot occur.
                        let _ = self.created_uuid.set(uuid.clone());
                        format!(" as {uuid}")
                    }
                    None => String::new(),
                };
                Ok(CommitOutcome::Reconciled {
                    succeeded: true,
                    job_id: None,
                    details: Some(format!(
                        "SDC {} completed{identity}; nothing reaches a device until a policy using it is deployed",
                        staged.action().as_str()
                    )),
                })
            }
            Err(SdcError::Cancelled | SdcError::Timeout | SdcError::MutationOutcomeUnknown) => {
                Ok(CommitOutcome::Indeterminate {
                    reason: "SDC IPS write was submitted but its outcome was not observed"
                        .to_owned(),
                })
            }
            Err(error) => {
                if error.is_definitive_rejection() {
                    self.refused_before_write.store(true, Ordering::SeqCst);
                }
                Err(error)
            }
        }
    }

    /// Release a planned-but-uncommitted IPS write.
    ///
    /// SDC has no candidate store and the coordinator only reaches `rollback`
    /// from pre-commit states, so nothing was written.
    async fn rollback(&self, to: RollbackRef) -> Result<RollbackOutcome, Self::Error> {
        match to {
            RollbackRef::CandidateRevert => Ok(RollbackOutcome {
                succeeded: true,
                details: Some(
                    "no remote candidate exists; SDC IPS writes are not staged".to_owned(),
                ),
            }),
            RollbackRef::Archive(_) | RollbackRef::Custom(_) => Err(SdcError::RollbackUnsupported),
        }
    }

    async fn unlock(&self) -> Result<UnlockOutcome, Self::Error> {
        Ok(UnlockOutcome::Released)
    }

    async fn confirm_commit(
        &self,
        _operation_id: &str,
        _attribution: &Attribution,
    ) -> Result<CommitOutcome, Self::Error> {
        Err(SdcError::InvalidInput(
            "SDC does not support confirmed IPS writes",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        IpsWriteAction, IpsWriteFamily, SdcPreparedIpsWrite, created_uuid_from,
        is_valid_profile_name,
    };
    use crate::ResourceKind;
    use serde_json::{Value, json};

    fn custom_signature() -> Value {
        json!({
            "uuid": "sig-1",
            "name": "CUSTOM-SQL-TIME-BASED",
            "signature_type": "signature",
            "severity": "major",
            "direction": "client-to-server"
        })
    }

    #[test]
    fn ips_families_read_and_write_the_same_collections() {
        for family in [IpsWriteFamily::Signature, IpsWriteFamily::Profile] {
            let kind = family.resource_kind().expect("flat family");
            assert_eq!(
                family.collection_segments(None),
                Some(kind.collection_segments().to_vec())
            );
            assert_eq!(family.collection_segments(Some("p-1")), None);
        }
        assert_eq!(
            IpsWriteFamily::Signature.resource_kind(),
            Some(ResourceKind::IpsSignatures)
        );
        assert_eq!(
            IpsWriteFamily::Profile.resource_kind(),
            Some(ResourceKind::IpsProfiles)
        );
        // A rule's collection is the one list_sdc_ips_rules reads.
        assert_eq!(IpsWriteFamily::Rule.resource_kind(), None);
        assert_eq!(IpsWriteFamily::Rule.collection_segments(None), None);
        assert_eq!(
            IpsWriteFamily::Rule.collection_segments(Some("p-1")),
            Some(vec!["api", "v1", "ips_profiles", "p-1", "ips_rules"])
        );
        assert_eq!(json!(IpsWriteFamily::Rule), json!("rule"));
    }

    fn rule_body() -> Value {
        json!({
            "name": "CLAUDE-TEST-266-RULE",
            "ips_signature": [
                { "uuid": "8ad65be9-3307-4763-b21e-5cf15b3f4653", "name": "CLAUDE-TEST-266-SIG" },
                { "uuid": "16955805-a9cd-4a5d-8d77-c6e5b03d558c", "name": "CLAUDE-TEST-266B-SIG" }
            ],
            "then": { "action": "no-action", "log_attacks": true },
            "position": 0
        })
    }

    #[test]
    fn a_rule_create_is_bound_to_its_profile() {
        let rule = SdcPreparedIpsWrite::new_rule(
            IpsWriteAction::Create,
            "ffc405a7-ef24-4a10-a6cb-0f8d4dfabfcc".to_owned(),
            None,
            rule_body(),
            Value::Null,
        )
        .expect("rule create");
        assert_eq!(rule.family(), IpsWriteFamily::Rule);
        assert_eq!(
            rule.profile_uuid(),
            Some("ffc405a7-ef24-4a10-a6cb-0f8d4dfabfcc")
        );
        assert_eq!(
            rule.plan()["profile_uuid"],
            "ffc405a7-ef24-4a10-a6cb-0f8d4dfabfcc"
        );
        assert_eq!(
            super::rule_signature_uuids(rule.request()),
            vec![
                "8ad65be9-3307-4763-b21e-5cf15b3f4653".to_owned(),
                "16955805-a9cd-4a5d-8d77-c6e5b03d558c".to_owned()
            ]
        );

        // The same rule in another profile is a different plan.
        let elsewhere = SdcPreparedIpsWrite::new_rule(
            IpsWriteAction::Create,
            "other-profile".to_owned(),
            None,
            rule_body(),
            Value::Null,
        )
        .expect("rule create");
        assert_ne!(rule.plan_digest(), elsewhere.plan_digest());

        // Moving a persisted rule to another profile breaks the digest.
        let mut stored = serde_json::to_value(&rule).expect("serialize");
        stored["profile_uuid"] = json!("other-profile");
        let moved: SdcPreparedIpsWrite = serde_json::from_value(stored).expect("deserialize");
        assert!(moved.validate().is_err());
    }

    #[test]
    fn flat_families_carry_no_profile_and_keep_their_plan_shape() {
        let signature = SdcPreparedIpsWrite::new(
            IpsWriteAction::Create,
            IpsWriteFamily::Signature,
            None,
            json!({ "name": "CUSTOM-X" }),
            Value::Null,
        )
        .expect("signature");
        assert!(signature.plan().get("profile_uuid").is_none());
        assert!(
            serde_json::to_value(&signature)
                .expect("serialize")
                .get("profile_uuid")
                .is_none(),
            "a signature envelope must serialize exactly as before rules existed"
        );
        // A rule cannot be built without its profile.
        assert!(
            SdcPreparedIpsWrite::new(
                IpsWriteAction::Create,
                IpsWriteFamily::Rule,
                None,
                rule_body(),
                Value::Null,
            )
            .is_err()
        );
        // A persisted signature envelope with a profile attached is refused.
        let mut stored = serde_json::to_value(&signature).expect("serialize");
        stored["profile_uuid"] = json!("p-1");
        let tampered: SdcPreparedIpsWrite = serde_json::from_value(stored).expect("deserialize");
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn a_rule_body_needs_signatures_and_an_action() {
        let refused = |body: Value| {
            SdcPreparedIpsWrite::new_rule(
                IpsWriteAction::Create,
                "p-1".to_owned(),
                None,
                body,
                Value::Null,
            )
            .is_err()
        };
        let without = |key: &str| {
            let mut body = rule_body();
            body.as_object_mut().expect("object").remove(key);
            body
        };
        assert!(refused(without("ips_signature")));
        assert!(refused(without("then")));
        for references in [
            json!([]),
            json!("8ad65be9"),
            json!([{ "name": "no uuid" }]),
            json!([{ "uuid": "../escape" }]),
            json!([{ "uuid": 42 }]),
        ] {
            let mut body = rule_body();
            body["ips_signature"] = references.clone();
            assert!(refused(body), "{references} accepted");
        }
        for then in [json!({}), json!({ "action": "" }), json!({ "action": 1 })] {
            let mut body = rule_body();
            body["then"] = then.clone();
            assert!(refused(body), "{then} accepted");
        }
        let mut body = rule_body();
        body["ips_signature"] = Value::Array(
            (0..257)
                .map(|index| json!({ "uuid": format!("sig-{index}") }))
                .collect(),
        );
        assert!(refused(body), "257 references accepted");
        assert!(
            SdcPreparedIpsWrite::new_rule(
                IpsWriteAction::Create,
                "../p".to_owned(),
                None,
                rule_body(),
                Value::Null,
            )
            .is_err(),
            "the parent profile is held to the identifier allowlist"
        );
    }

    #[test]
    fn a_custom_signature_create_and_update_are_accepted() {
        let created = SdcPreparedIpsWrite::new(
            IpsWriteAction::Create,
            IpsWriteFamily::Signature,
            None,
            json!({ "name": "CUSTOM-X", "signature_type": "signature" }),
            Value::Null,
        )
        .expect("custom create");
        assert_eq!(created.operation(), "ips_write");
        assert!(created.plan_digest().starts_with("sha256:"));

        SdcPreparedIpsWrite::new(
            IpsWriteAction::Update,
            IpsWriteFamily::Signature,
            Some("sig-1".to_owned()),
            json!({ "name": "CUSTOM-SQL-TIME-BASED", "severity": "critical" }),
            custom_signature(),
        )
        .expect("custom update");
    }

    #[test]
    fn a_predefined_signature_is_never_updated() {
        let mut predefined = custom_signature();
        predefined["import_id"] = json!("HTTP:SQL:INJ:COMMENT");
        let error = SdcPreparedIpsWrite::new(
            IpsWriteAction::Update,
            IpsWriteFamily::Signature,
            Some("sig-1".to_owned()),
            json!({ "name": "HTTP:SQL:INJ:COMMENT", "severity": "info" }),
            predefined,
        )
        .expect_err("predefined update must be refused");
        assert!(error.to_string().contains("predefined"));
    }

    #[test]
    fn an_activation_date_also_marks_a_signature_predefined() {
        for marker in [
            json!("2026-09-01T00:00:00Z"),
            json!(1_756_684_800),
            json!(true),
        ] {
            let mut bundled = custom_signature();
            bundled["activation_date"] = marker.clone();
            assert!(
                SdcPreparedIpsWrite::new(
                    IpsWriteAction::Update,
                    IpsWriteFamily::Signature,
                    Some("sig-1".to_owned()),
                    json!({ "name": "CUSTOM-SQL-TIME-BASED" }),
                    bundled,
                )
                .is_err(),
                "activation_date {marker} must mark the signature predefined"
            );
        }
    }

    #[test]
    fn an_update_needs_a_known_observed_signature_type() {
        for observed in [Value::Null, json!("custom"), json!(""), json!(7)] {
            let mut before = custom_signature();
            if observed.is_null() {
                before
                    .as_object_mut()
                    .expect("object")
                    .remove("signature_type");
            } else {
                before["signature_type"] = observed.clone();
            }
            let error = SdcPreparedIpsWrite::new(
                IpsWriteAction::Update,
                IpsWriteFamily::Signature,
                Some("sig-1".to_owned()),
                json!({ "name": "CUSTOM-SQL-TIME-BASED" }),
                before,
            )
            .expect_err("unknown observed type must fail closed");
            assert!(
                error.to_string().contains("signature_type"),
                "{observed}: {error}"
            );
        }
    }

    #[test]
    fn a_body_signature_type_must_be_one_the_spec_allows() {
        for kind in ["anomaly", "signature", "chain", "static", "dynamic"] {
            SdcPreparedIpsWrite::new(
                IpsWriteAction::Create,
                IpsWriteFamily::Signature,
                None,
                json!({ "name": "CUSTOM-X", "signature_type": kind }),
                Value::Null,
            )
            .expect("spec value accepted");
        }
        for kind in [json!("custom"), json!("Signature"), json!(1)] {
            assert!(
                SdcPreparedIpsWrite::new(
                    IpsWriteAction::Create,
                    IpsWriteFamily::Signature,
                    None,
                    json!({ "name": "CUSTOM-X", "signature_type": kind }),
                    Value::Null,
                )
                .is_err(),
                "{kind} accepted"
            );
        }
    }

    #[test]
    fn identifiers_are_held_to_a_strict_allowlist() {
        assert!(super::is_allowed_identifier(
            "8ad65be9-3307-4763-b21e-5cf15b3f4653"
        ));
        assert!(super::is_allowed_identifier("abc_DEF:1+2="));
        for bad in [
            "", "../sig", "sig/1", "sig%2F1", "sig.1", "sig 1", "sig\n1", "sigé",
        ] {
            assert!(!super::is_allowed_identifier(bad), "{bad:?} accepted");
        }
        assert!(!super::is_allowed_identifier(&"a".repeat(257)));
        assert!(
            SdcPreparedIpsWrite::new(
                IpsWriteAction::Update,
                IpsWriteFamily::Profile,
                Some("p/../1".to_owned()),
                json!({ "name": "P" }),
                json!({ "uuid": "p", "name": "P" }),
            )
            .is_err()
        );
    }

    #[test]
    fn an_empty_import_id_does_not_mark_a_signature_predefined() {
        let mut custom = custom_signature();
        custom["import_id"] = json!("  ");
        SdcPreparedIpsWrite::new(
            IpsWriteAction::Update,
            IpsWriteFamily::Signature,
            Some("sig-1".to_owned()),
            json!({ "name": "CUSTOM-SQL-TIME-BASED" }),
            custom,
        )
        .expect("blank import_id is custom");
    }

    #[test]
    fn a_body_cannot_set_server_assigned_fields() {
        for field in ["uuid", "import_id"] {
            let mut body = json!({ "name": "CUSTOM-X" });
            body[field] = json!("forged");
            let error = SdcPreparedIpsWrite::new(
                IpsWriteAction::Create,
                IpsWriteFamily::Signature,
                None,
                body,
                Value::Null,
            )
            .expect_err("server-assigned field must be refused");
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn a_write_must_match_the_shape_of_its_action() {
        // Create with a UUID.
        assert!(
            SdcPreparedIpsWrite::new(
                IpsWriteAction::Create,
                IpsWriteFamily::Profile,
                Some("p-1".to_owned()),
                json!({ "name": "P" }),
                Value::Null,
            )
            .is_err()
        );
        // Update without a UUID.
        assert!(
            SdcPreparedIpsWrite::new(
                IpsWriteAction::Update,
                IpsWriteFamily::Profile,
                None,
                json!({ "name": "P" }),
                json!({ "uuid": "p-1", "name": "P" }),
            )
            .is_err()
        );
        // Update without observed before-state.
        assert!(
            SdcPreparedIpsWrite::new(
                IpsWriteAction::Update,
                IpsWriteFamily::Profile,
                Some("p-1".to_owned()),
                json!({ "name": "P" }),
                Value::Null,
            )
            .is_err()
        );
        // Empty or non-object bodies, and a missing name.
        for body in [
            json!({}),
            json!([]),
            json!("P"),
            json!({ "description": "x" }),
        ] {
            assert!(
                SdcPreparedIpsWrite::new(
                    IpsWriteAction::Create,
                    IpsWriteFamily::Profile,
                    None,
                    body,
                    Value::Null,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn profile_names_follow_the_spec_format() {
        assert!(is_valid_profile_name("PROTECT-DVWA"));
        assert!(is_valid_profile_name(&"a".repeat(127)));
        assert!(!is_valid_profile_name(""));
        assert!(!is_valid_profile_name(&"a".repeat(128)));
        for bad in ["has space", "a#b", "a&b", "a;b", "a\rb", "a\nb"] {
            assert!(!is_valid_profile_name(bad), "{bad:?} accepted");
        }
        assert!(
            SdcPreparedIpsWrite::new(
                IpsWriteAction::Create,
                IpsWriteFamily::Profile,
                None,
                json!({ "name": "bad name" }),
                Value::Null,
            )
            .is_err()
        );
    }

    #[test]
    fn a_tampered_envelope_fails_its_digest() {
        let prepared = SdcPreparedIpsWrite::new(
            IpsWriteAction::Create,
            IpsWriteFamily::Signature,
            None,
            json!({ "name": "CUSTOM-X", "severity": "minor" }),
            Value::Null,
        )
        .expect("prepared");
        let mut stored = serde_json::to_value(&prepared).expect("serialize");
        stored["request"]["severity"] = json!("critical");
        let tampered: SdcPreparedIpsWrite = serde_json::from_value(stored).expect("deserialize");
        let error = tampered.validate().expect_err("digest mismatch");
        assert!(error.to_string().contains("digest"));
    }

    #[test]
    fn a_tampered_before_state_cannot_hide_a_predefined_signature() {
        // A predefined signature planned as custom by editing the stored
        // before-state also breaks the digest, so the guard cannot be bypassed
        // through the state file.
        let prepared = SdcPreparedIpsWrite::new(
            IpsWriteAction::Update,
            IpsWriteFamily::Signature,
            Some("sig-1".to_owned()),
            json!({ "name": "CUSTOM-SQL-TIME-BASED" }),
            custom_signature(),
        )
        .expect("prepared");
        let mut stored = serde_json::to_value(&prepared).expect("serialize");
        stored["before"]["import_id"] = json!("HTTP:SQL:INJ:COMMENT");
        let tampered: SdcPreparedIpsWrite = serde_json::from_value(stored).expect("deserialize");
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn a_create_response_yields_the_allocated_uuid_when_present() {
        assert_eq!(
            created_uuid_from(&json!({ "uuid": "8ad65be9-0000", "name": "X" })).as_deref(),
            Some("8ad65be9-0000")
        );
        for response in [
            Value::Null,
            json!({}),
            json!({ "uuid": "" }),
            json!({ "uuid": "  " }),
            json!({ "uuid": 42 }),
            json!([{ "uuid": "nested" }]),
        ] {
            assert_eq!(created_uuid_from(&response), None, "{response}");
        }
    }

    #[test]
    fn the_wire_names_are_stable() {
        assert_eq!(json!(IpsWriteFamily::Signature), json!("signature"));
        assert_eq!(json!(IpsWriteFamily::Profile), json!("profile"));
        assert_eq!(json!(IpsWriteAction::Create), json!("create"));
        assert_eq!(json!(IpsWriteAction::Update), json!("update"));
    }
}
