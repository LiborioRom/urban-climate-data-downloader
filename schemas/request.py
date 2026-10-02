from __future__ import annotations

from datetime import date
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ROISpecification(str, Enum):
    PROJECT_DEFAULT = "project_default"
    CENTER_AND_HALF_SIDE = "center_and_half_side"
    BBOX = "bbox"


class OutputFormat(str, Enum):
    CSV = "csv"


class DatasetPurpose(str, Enum):
    GENERAL_DOWNLOAD = "general_download"
    LST_DOWNSCALING = "lst_downscaling"


class DateSelectionMode(str, Enum):
    CONTINUOUS = "continuous"
    RECURRING_WINDOW = "recurring_window"


class DateWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_order(self) -> "DateWindow":
        if self.start_date > self.end_date:
            raise ValueError("La fecha inicial de una ventana es posterior a la final")
        return self


class SourcePreference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable: str
    source: str


class ROIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specification: ROISpecification
    area_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    area_name: str | None = Field(default=None, max_length=120)
    place_name: str | None = None
    center_lat: float | None = Field(default=None, ge=-90, le=90)
    center_lon: float | None = Field(default=None, ge=-180, le=180)
    half_side_m: float | None = Field(default=None, gt=0)
    rotation_deg: float = Field(default=0.0, ge=0, lt=90)
    bbox: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "ROIRequest":
        if self.specification == ROISpecification.CENTER_AND_HALF_SIDE:
            if self.center_lat is None or self.center_lon is None or self.half_side_m is None:
                raise ValueError("center_and_half_side requiere center_lat, center_lon y half_side_m")
        if self.specification == ROISpecification.BBOX and self.bbox is None:
            raise ValueError("bbox requiere [min_lon, min_lat, max_lon, max_lat]")
        if self.bbox is not None:
            min_lon, min_lat, max_lon, max_lat = self.bbox
            if min_lon >= max_lon or min_lat >= max_lat:
                raise ValueError("El bbox no tiene límites válidos")
        return self


class PipelineParameters(BaseModel):
    """Parámetros ya presentes en el notebook; sus valores por defecto no cambian el pipeline."""

    model_config = ConfigDict(extra="forbid")

    allowed_months: list[int] = Field(default_factory=lambda: [5, 6, 7, 8, 9, 10])
    max_landsat_cloud: int = Field(default=80, ge=0, le=100)
    max_sentinel2_cloud: int = Field(default=80, ge=0, le=100)
    s2_max_day_difference: int = Field(default=20, ge=0)
    s3_max_day_difference: int = Field(default=3, ge=0)
    include_urban_morphology: bool = True
    overwrite_outputs: bool = False
    max_parallel_areas: int = Field(default=2, ge=1, le=2)


class DatasetRequest(BaseModel):
    """Petición parcial: el motor de reglas decide qué campos faltan antes de ejecutar."""

    model_config = ConfigDict(extra="forbid")

    roi: ROIRequest | None = None
    areas: list[ROIRequest] = Field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None
    date_selection_mode: DateSelectionMode = DateSelectionMode.CONTINUOUS
    date_windows: list[DateWindow] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    target_resolution_m: float | None = Field(default=None, gt=0)
    preferred_sources: list[SourcePreference] = Field(default_factory=list)
    crs: str = "EPSG:25830"
    output_format: OutputFormat = OutputFormat.CSV
    dataset_purpose: DatasetPurpose = DatasetPurpose.GENERAL_DOWNLOAD
    parameters: PipelineParameters = Field(default_factory=PipelineParameters)
    user_notes: str | None = None
