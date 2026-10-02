from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .request import DatasetRequest


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Severity
    code: str
    message: str
    variable: str | None = None
    source: str | None = None


class ValidationReport(BaseModel):
    request: DatasetRequest
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == Severity.ERROR for issue in self.issues)


class VariablePlan(BaseModel):
    variable: str
    source: str
    tool: str
    dataset_or_product: str | None = None
    bands: list[str] = Field(default_factory=list)
    native_resolution_m: float | None = None
    target_resolution_m: float
    transformations: list[str] = Field(default_factory=list)
    execution_supported: bool = True


class PlanStep(BaseModel):
    order: int
    tool: str
    action: str
    variables: list[str] = Field(default_factory=list)


class AcquisitionPlan(BaseModel):
    request: DatasetRequest
    variable_plans: list[VariablePlan]
    steps: list[PlanStep]
    issues: list[ValidationIssue]
    harmonization_steps: list[str] = Field(default_factory=list)
    execution_profile: str = "current_notebook_full_pipeline"

    @property
    def executable(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)

