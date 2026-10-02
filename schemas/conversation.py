from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .request import DatasetRequest


class Intent(str, Enum):
    CREATE_OR_UPDATE_REQUEST = "create_or_update_request"
    DEFINE_ROI = "define_roi"
    MOVE_ROI = "move_roi"
    RESIZE_ROI = "resize_roi"
    UNDO_ROI = "undo_roi"
    CONFIRM_ROI = "confirm_roi"
    EXECUTE = "execute"
    ASK_QUESTION = "ask_question"
    CLARIFY = "clarify"


class RequestPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date | None = None
    end_date: date | None = None
    variables: list[str] | None = None
    variable_action: Literal["replace", "add", "remove"] | None = None
    target_resolution_m: float | None = Field(default=None, gt=0)


class ROIDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    place_query: str | None = None
    geometry: Literal["square"] = "square"
    area_km2: float | None = Field(default=None, gt=0)
    half_side_m: float | None = Field(default=None, gt=0)
    spatial_relation: Literal["around", "centered_on", "between"] = "around"
    reference_places: list[str] = Field(default_factory=list)


class ROIAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Literal["north", "south", "east", "west"] | None = None
    distance_m: float | None = Field(default=None, gt=0)
    scale_factor: float | None = Field(default=None, gt=0)


class AgentTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    request_patch: RequestPatch | None = None
    roi_definition: ROIDefinition | None = None
    roi_adjustment: ROIAdjustment | None = None
    assistant_message: str | None = None


class ClarificationAnswer(BaseModel):
    """Aclaración conversacional generada sin autorizar cambios en la solicitud."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)


class VariableKnowledgeAnswer(BaseModel):
    """Respuesta de Qwen limitada a la evidencia recuperada del catálogo."""

    model_config = ConfigDict(extra="forbid")

    answerable_from_cards: bool
    answer: str = Field(min_length=1)
    referenced_variables: list[str] = Field(default_factory=list)
    missing_information: str | None = None

    @model_validator(mode="after")
    def require_missing_information_when_unsupported(self):
        if not self.answerable_from_cards and not self.missing_information:
            raise ValueError("Una respuesta no respaldada debe indicar qué información falta.")
        return self


class VariableCatalogueSelection(BaseModel):
    """Selección temática previa realizada por Qwen sobre el índice del catálogo."""

    model_config = ConfigDict(extra="forbid")

    selected_variables: list[str] = Field(default_factory=list, max_length=10)


class ROIState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center_lat: float
    center_lon: float
    half_side_m: float = Field(gt=0)
    rotation_deg: float = Field(default=0.0, ge=0, lt=90)
    place_name: str | None = None
    display_name: str | None = None
    osm_type: str | None = None
    osm_id: int | None = None
    boundary_geojson: dict[str, Any] | None = None
    coverage_percent: float | None = None
    confirmed: bool = False
    provenance: str = "user_coordinates"

    @property
    def area_km2(self) -> float:
        return (2 * self.half_side_m) ** 2 / 1_000_000


class ConversationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: DatasetRequest = Field(default_factory=DatasetRequest)
    active_roi: ROIState | None = None
    roi_history: list[ROIState] = Field(default_factory=list)
    last_intent: Intent | None = None


class DialogueResult(BaseModel):
    state: ConversationState
    assistant_message: str
    show_map: bool = False
    execute_requested: bool = False
    provider: str
