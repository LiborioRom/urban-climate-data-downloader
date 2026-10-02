from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPLANATIONS = {
    2: "Este bloque instala las dependencias científicas y geoespaciales necesarias en el kernel local. Se ejecuta una sola vez y no altera las decisiones metodológicas del pipeline.",
    4: "Aquí se define el experimento: periodo, ROI, mallas, controles de calidad y salidas. Estos parámetros determinan el dominio espacial y temporal de todas las adquisiciones posteriores.",
    6: "Este bloque obtiene credenciales exclusivamente del entorno local. Los secretos habilitan los servicios externos, pero no forman parte de los datos científicos ni deben persistirse en resultados.",
    8: "Se cargan las librerías, se validan los parámetros y se construyen las fechas efectivas. Esta etapa fija las invariantes antes de cualquier consulta remota.",
    10: "El ROI se transforma a una proyección métrica estable y se discretiza en nodos. Trabajar en EPSG:25830 permite expresar distancias y áreas de forma coherente para Sevilla.",
    12: "Se autentican los servicios de datos y se comprueba su disponibilidad. La autenticación es infraestructura; no modifica los algoritmos científicos siguientes.",
}


def code_digest(cells: list[dict]) -> str:
    payload = [cell.get("source", []) for cell in cells if cell.get("cell_type") == "code"]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def document(source: Path, target: Path) -> None:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    original_digest = code_digest(notebook["cells"])
    documented = []
    code_counter = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            text = EXPLANATIONS.get(index) or (
                "Este bloque aplica la siguiente etapa lógica del pipeline científico existente. "
                "Sus entradas proceden de los bloques anteriores y sus resultados alimentan la armonización o salida posterior."
            )
            documented.append({"cell_type": "markdown", "metadata": {"generated_documentation": True}, "source": [f"**Propósito científico.** {text}\n"]})
            code_counter += 1
        documented.append(cell)
    notebook["cells"] = documented
    if code_digest(notebook["cells"]) != original_digest:
        raise RuntimeError("La documentación alteró alguna celda de código.")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Documentadas {code_counter} celdas de código sin modificar su contenido.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    document(args.source, args.target)
