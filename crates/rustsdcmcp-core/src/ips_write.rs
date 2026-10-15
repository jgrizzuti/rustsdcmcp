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
}

impl IpsWriteFamily {
    /// Every writable IPS family, for exhaustive iteration in tests.
    pub const ALL: &'static [Self] = &[Self::Signature, Self::Profile];

    /// The read family used for before-state and drift detection.
    #[must_use]
    pub const fn resource_kind(self) -> ResourceKind {
        match self {
            Self::Signature => ResourceKind::IpsSignatures,
            Self::Profile => ResourceKind::IpsProfiles,
        }
    }

    /// Exact collection path segments, delegated to the read catalog so a
    /// write and its drift-detection read can never address different paths.
    #[must_use]
    pub const fn collection_segments(self) -> &'static [&'static str] {
        self.resource_kind().collection_segments()
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
    uuid: Option<String>,
    request: Value,
    before: Value,
    plan_digest: String,
}

impl SdcPreparedIpsWrite {
    /// Build a canonical, digest-bound IPS write.
    ///
    /// `before` is the target's observed state for update, and `Value::Null`
    /// for create.
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
        let plan = plan_artifact(action, family, uuid.as_deref(), &request, &before);
        let prepared = Self {
            operation: "ips_write".to_owned(),
            action,
            family,
            uuid,
            request,
            before,
            plan_digest: canonical_digest(&plan)
                .map_err(|error| SdcError::PreparedChange(error.to_string()))?,
        };
        prepared.validate()?;
        Ok(prepared)
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
            && (uuid.is_empty()
                || uuid.len() > 256
                || uuid
                    .bytes()
                    .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace()))
        {
            return Err(SdcError::PreparedChange(
                "prepared IPS UUID must be 1-256 non-whitespace bytes".to_owned(),
            ));
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
        if self.family == IpsWriteFamily::Profile && !is_valid_profile_name(name) {
            return Err(SdcError::PreparedChange(
                "IPS profile name must be 1-127 characters with no space, `#`, `&`, `;`, CR or LF"
                    .to_owned(),
            ));
        }
        if self.family == IpsWriteFamily::Signature && is_predefined_signature(&self.before) {
            return Err(SdcError::PreparedChange(
                "refusing to update a predefined signature imported from the signature \
                 bundle; only custom signatures are writable"
                    .to_owned(),
            ));
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

/// Whether an observed signature came from Juniper's signature bundle.
///
/// The vendored spec documents `import_id` as "applicable only for predefined
/// signatures imported from signature bundles, not for custom signatures or
/// groups". It is the only documented discriminator: `signature_type` is the
/// detection kind (`anomaly`, `signature`, `chain`, `static`, `dynamic`), not
/// its origin. Not yet confirmed against a live tenant.
fn is_predefined_signature(before: &Value) -> bool {
    before
        .get("import_id")
        .and_then(Value::as_str)
        .is_some_and(|import_id| !import_id.trim().is_empty())
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
    uuid: Option<&str>,
    request: &Value,
    before: &Value,
) -> Value {
    json!({
        "action": action,
        "family": family,
        "uuid": uuid,
        "before": before,
        "after": request,
    })
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
            .get_resource(staged.family().resource_kind(), uuid, &self.cancellation)
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
                    .create_ips_object(family, staged.request(), &self.cancellation)
                    .await
            }
            (IpsWriteAction::Update, Some(uuid)) => {
                self.client
                    .update_ips_object(family, uuid, staged.request(), &self.cancellation)
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
        for family in IpsWriteFamily::ALL {
            assert_eq!(
                family.collection_segments(),
                family.resource_kind().collection_segments()
            );
        }
        assert_eq!(
            IpsWriteFamily::Signature.resource_kind(),
            ResourceKind::IpsSignatures
        );
        assert_eq!(
            IpsWriteFamily::Profile.resource_kind(),
            ResourceKind::IpsProfiles
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
