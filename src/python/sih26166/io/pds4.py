# [ANNOTATION] Module docstring describing PDS4 XML label parser for Chandrayaan-2 metadata ingestion.
"""
PDS4 metadata ingestion for SIH26166.

Supports the PDS4 label structure used by Chandrayaan-2 products
distributed by ISRO/ISSDC.

This module reads metadata only. It does NOT load the associated
binary image into memory.
"""

# [ANNOTATION] Enable future type hint syntax.
from __future__ import annotations

# [ANNOTATION] Import standard library dataclass primitives, XML parser, and type hints.
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any


# [ANNOTATION] Dataclass storing metadata fields parsed from a PDS4 observational image XML label.
@dataclass(frozen=True)
class PDS4ImageMetadata:
    """Metadata extracted from a PDS4 observational image label."""

    label_path: Path

    # Product identification
    product_id: str | None = None
    logical_identifier: str | None = None
    version_id: str | None = None
    title: str | None = None
    product_class: str | None = None

    # Observation
    start_time: str | None = None
    stop_time: str | None = None
    purpose: str | None = None
    processing_level: str | None = None
    processing_description: str | None = None

    # Observing system
    mission_name: str | None = None
    spacecraft_name: str | None = None
    instrument_name: str | None = None
    instrument_type: str | None = None

    # Image file
    file_name: str | None = None
    file_size_bytes: int | None = None
    md5_checksum: str | None = None
    offset_bytes: int | None = None

    # Array
    axes: int | None = None
    axis_index_order: str | None = None
    width: int | None = None
    height: int | None = None
    bands: int | None = None
    data_type: str | None = None
    bit_depth: int | None = None
    signed: bool | None = None

    # Axis metadata
    line_elements: int | None = None
    sample_elements: int | None = None

    # Optional instrument metadata
    pixel_size_microns: float | None = None
    focal_length_mm: float | None = None
    exposure_time_ms: float | None = None

    # Optional geometry
    upper_left_latitude: float | None = None
    upper_left_longitude: float | None = None
    upper_right_latitude: float | None = None
    upper_right_longitude: float | None = None
    lower_left_latitude: float | None = None
    lower_left_longitude: float | None = None
    lower_right_latitude: float | None = None
    lower_right_longitude: float | None = None

    # Lightweight raw summary for debugging/reproducibility
    raw: dict[str, Any] | None = None


# [ANNOTATION] Helper function extracting local name from an XML namespace tag.
def _local_name(tag: str) -> str:
    """Return an XML tag's local name, ignoring its namespace."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


# [ANNOTATION] Helper function retrieving direct child elements matching local tag name.
def _children(element: ET.Element, name: str) -> list[ET.Element]:
    """Return direct children matching a namespace-independent name."""
    return [
        child
        for child in list(element)
        if _local_name(child.tag) == name
    ]


# [ANNOTATION] Helper function retrieving first direct child element matching tag name.
def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    """Return the first direct child matching a namespace-independent name."""
    if element is None:
        return None

    for child in list(element):
        if _local_name(child.tag) == name:
            return child

    return None


# [ANNOTATION] Helper function retrieving all descendant nodes matching tag name.
def _descendants(element: ET.Element, name: str) -> list[ET.Element]:
    """Return descendants matching a namespace-independent name."""
    return [
        node
        for node in element.iter()
        if _local_name(node.tag) == name
    ]


# [ANNOTATION] Helper function retrieving first descendant matching tag name.
def _first(element: ET.Element | None, name: str) -> ET.Element | None:
    """Return the first descendant matching a namespace-independent name."""
    if element is None:
        return None

    for node in element.iter():
        if _local_name(node.tag) == name:
            return node

    return None


# [ANNOTATION] Helper function stripping whitespace and extracting text content from an element.
def _text(element: ET.Element | None) -> str | None:
    """Return normalized element text."""
    if element is None or element.text is None:
        return None

    value = element.text.strip()
    return value if value else None


# [ANNOTATION] Helper function extracting text value of named child element.
def _value(element: ET.Element | None, name: str) -> str | None:
    """Return text of a named descendant."""
    return _text(_first(element, name))


# [ANNOTATION] Helper function converting string input into integer value safely.
def _int_value(value: str | None) -> int | None:
    """Convert a textual integer to int."""
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


# [ANNOTATION] Helper function converting string input into floating point value safely.
def _float_value(value: str | None) -> float | None:
    """Convert a textual number to float."""
    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


# [ANNOTATION] Helper function determining signedness based on PDS4 data type string.
def _bool_from_data_type(data_type: str | None) -> bool | None:
    """Infer signedness when the PDS4 data type explicitly encodes it."""
    if data_type is None:
        return None

    normalized = data_type.lower()

    if "unsigned" in normalized:
        return False

    if "signed" in normalized:
        return True

    return None


# [ANNOTATION] Helper function mapping PDS4 data type string to numerical bit depth.
def _bits_from_data_type(data_type: str | None) -> int | None:
    """Infer common bit depth from a PDS4 data type."""
    if data_type is None:
        return None

    normalized = data_type.lower()

    mapping = {
        "unsignedbyte": 8,
        "signedbyte": 8,
        "unsignedinteger": 16,
        "signedinteger": 16,
        "unsignedlsb2": 16,
        "signedlsb2": 16,
        "unsignedmsb2": 16,
        "signedmsb2": 16,
        "unsignedmsbin": 16,
        "signedmsbin": 16,
        "unsignedlsbin": 16,
        "signedlsbin": 16,
        "unsignedlsb4": 32,
        "signedlsb4": 32,
        "unsignedmsb4": 32,
        "signedmsb4": 32,
        "unsignedlong": 32,
        "signedlong": 32,
        "unsignedmsb": 32,
        "signedmsb": 32,
        "unsignedlsb": 32,
        "signedlsb": 32,
        "real": 32,
        "double": 64,
    }

    return mapping.get(normalized)


# [ANNOTATION] Helper function finding geometry parameter values in XML tree.
def _find_isda_value(
    root: ET.Element,
    local_name: str,
) -> float | None:
    """
    Find an ISDA geometry value.

    Namespace prefixes are deliberately ignored because PDS4 labels
    can use different namespace declarations while retaining the same
    local element names.
    """
    element = _first(root, local_name)
    return _float_value(_text_element(element))


def _text_element(element: ET.Element | None) -> str | None:
    return _text(element)


# [ANNOTATION] Helper function finding specific Axis_Array element by name.
def _find_axis(array: ET.Element, axis_name: str) -> ET.Element | None:
    """Find an Axis_Array with the requested axis_name."""
    for axis in _children(array, "Axis_Array"):
        name = _value(axis, "axis_name")
        if name and name.lower() == axis_name.lower():
            return axis

    return None


# [ANNOTATION] Helper function extracting product ID string from ISRO logical identifier string.
def _extract_product_id(logical_identifier: str | None) -> str | None:
    """
    Extract the product identifier from an ISRO logical identifier.

    Example:
        urn:isro:isda:ch2_cho.ohr:data_raw:
        ch2_ohr_nrp_20200824t1003365280_d_img_d18
    """
    if not logical_identifier:
        return None

    return logical_identifier.rsplit(":", 1)[-1]


# [ANNOTATION] Primary function parsing PDS4 XML label to construct PDS4ImageMetadata dataclass instance.
def read_pds4_label(label_path: str | Path) -> PDS4ImageMetadata:
    """
    Read a PDS4 XML label and return image metadata.

    The associated binary image is never opened or loaded.

    Parameters
    ----------
    label_path:
        Path to a PDS4 XML label.

    Returns
    -------
    PDS4ImageMetadata
        Parsed metadata.

    Raises
    ------
    FileNotFoundError
        If the label does not exist.
    ValueError
        If the XML cannot be parsed or is not a PDS4 product label.
    """
    path = Path(label_path)
    print(f"[PDS4] Ingesting PDS4 XML label metadata from: {path.name}...")

    if not path.is_file():
        raise FileNotFoundError(f"PDS4 label not found: {path}")

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML/PDS4 label: {path}") from exc

    root = tree.getroot()

    if _local_name(root.tag) != "Product_Observational":
        raise ValueError(
            f"Unsupported PDS4 product type: {_local_name(root.tag)!r}"
        )

    identification = _first(root, "Identification_Area")
    observation = _first(root, "Observation_Area")
    file_area = _first(root, "File_Area_Observational")
    array = _first(file_area, "Array_2D_Image")

    logical_identifier = _value(
        identification,
        "logical_identifier",
    )

    version_id = _value(
        identification,
        "version_id",
    )

    title = _value(
        identification,
        "title",
    )

    product_class = _value(
        identification,
        "product_class",
    )

    # ------------------------------------------------------------------
    # Observation metadata
    # ------------------------------------------------------------------

    time_coordinates = _first(observation, "Time_Coordinates")

    start_time = _value(
        time_coordinates,
        "start_date_time",
    )

    stop_time = _value(
        time_coordinates,
        "stop_date_time",
    )

    primary_result = _first(
        observation,
        "Primary_Result_Summary",
    )

    purpose = _value(primary_result, "purpose")
    processing_level = _value(primary_result, "processing_level")
    processing_description = _value(primary_result, "description")

    investigation_area = _first(
        observation,
        "Investigation_Area",
    )

    investigation_name = _value(
        investigation_area,
        "name",
    )

    investigation_type = _value(
        investigation_area,
        "type",
    )

    mission_name: str | None = None

    if investigation_type and investigation_type.lower() == "mission":
        mission_name = investigation_name

    # ------------------------------------------------------------------
    # Observing system
    # ------------------------------------------------------------------
    
    observing_system = _first(
        observation,
        "Observing_System",
    )

    components = _children(
        observing_system, # type: ignore
        "Observing_System_Component",
    )

    spacecraft_name: str | None = None
    instrument_name: str | None = None
    instrument_type: str | None = None

    for component in components:
        name = _value(component, "name")
        component_type = _value(component, "type")

        if component_type:
            normalized_type = component_type.lower()

            if normalized_type == "mission":
                mission_name = name

            elif normalized_type == "spacecraft":
                spacecraft_name = name

            elif normalized_type == "instrument":
                instrument_name = name
                instrument_type = component_type

    # ------------------------------------------------------------------
    # File metadata
    # ------------------------------------------------------------------

    file_element = _first(file_area, "File")

    file_name = _value(file_element, "file_name")
    file_size_bytes = _int_value(
        _value(file_element, "file_size")
    )
    md5_checksum = _value(
        file_element,
        "md5_checksum",
    )

    # ------------------------------------------------------------------
    # Array metadata
    # ------------------------------------------------------------------

    offset_bytes = _int_value(
        _value(array, "offset")
    )

    axes = _int_value(
        _value(array, "axes")
    )

    axis_index_order = _value(
        array,
        "axis_index_order",
    )

    element_array = _first(
        array,
        "Element_Array",
    )

    data_type = _value(
        element_array,
        "data_type",
    )

    bit_depth = _bits_from_data_type(data_type)
    signed = _bool_from_data_type(data_type)

    line_axis = _find_axis(array, "Line") # type: ignore
    sample_axis = _find_axis(array, "Sample") # type: ignore

    line_elements = _int_value(
        _value(line_axis, "elements")
    )

    sample_elements = _int_value(
        _value(sample_axis, "elements")
    )

    height = line_elements
    width = sample_elements

    # ------------------------------------------------------------------
    # Optional instrument parameters
    # ------------------------------------------------------------------

    pixel_size_microns = _float_value(
        _value(root, "pixel_size")
    )

    focal_length_mm = _float_value(
        _value(root, "focal_length")
    )

    exposure_time_ms = _float_value(
        _value(root, "exposure_time")
    )

    # ------------------------------------------------------------------
    # Optional ISDA geometry
    # ------------------------------------------------------------------

    upper_left_latitude = _find_isda_value(root, "upper_left_latitude")
    upper_left_longitude = _find_isda_value(root, "upper_left_longitude")
    upper_right_latitude = _find_isda_value(root, "upper_right_latitude")
    upper_right_longitude = _find_isda_value(root, "upper_right_longitude")
    lower_left_latitude = _find_isda_value(root, "lower_left_latitude")
    lower_left_longitude = _find_isda_value(root, "lower_left_longitude")
    lower_right_latitude = _find_isda_value(root, "lower_right_latitude")
    lower_right_longitude = _find_isda_value(root, "lower_right_longitude")

    raw_summary: dict[str, Any] = {
        "root_tag": _local_name(root.tag),
        "namespace": (
            root.tag.split("}", 1)[0][1:]
            if root.tag.startswith("{")
            else None
        ),
        "axis_names": [
            _value(axis, "axis_name")
            for axis in _children(array, "Axis_Array")
        ]
        if array is not None
        else [],
    }

    print(f"[PDS4] Parsed product metadata: ID='{_extract_product_id(logical_identifier)}', Dimensions={width}x{height}.")

    return PDS4ImageMetadata(
        label_path=path.resolve(),
        product_id=_extract_product_id(logical_identifier),
        logical_identifier=logical_identifier,
        version_id=version_id,
        title=title,
        product_class=product_class,
        start_time=start_time,
        stop_time=stop_time,
        purpose=purpose,
        processing_level=processing_level,
        processing_description=processing_description,
        mission_name=mission_name,
        spacecraft_name=spacecraft_name,
        instrument_name=instrument_name,
        instrument_type=instrument_type,
        file_name=file_name,
        file_size_bytes=file_size_bytes,
        md5_checksum=md5_checksum,
        offset_bytes=offset_bytes,
        axes=axes,
        axis_index_order=axis_index_order,
        width=width,
        height=height,
        bands=1,
        data_type=data_type,
        bit_depth=bit_depth,
        signed=signed,
        line_elements=line_elements,
        sample_elements=sample_elements,
        pixel_size_microns=pixel_size_microns,
        focal_length_mm=focal_length_mm,
        exposure_time_ms=exposure_time_ms,
        upper_left_latitude=upper_left_latitude,
        upper_left_longitude=upper_left_longitude,
        upper_right_latitude=upper_right_latitude,
        upper_right_longitude=upper_right_longitude,
        lower_left_latitude=lower_left_latitude,
        lower_left_longitude=lower_left_longitude,
        lower_right_latitude=lower_right_latitude,
        lower_right_longitude=lower_right_longitude,
        raw=raw_summary,
    )