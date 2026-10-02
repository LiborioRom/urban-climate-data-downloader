from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UserFacingError:
    code: str
    title: str
    explanation: str
    recovery_steps: tuple[str, ...]
    technical_type: str

    def as_chat_message(self) -> str:
        steps = "\n".join(f"{index}. {step}" for index, step in enumerate(self.recovery_steps, 1))
        return f"**{self.title}**\n\n{self.explanation}\n\n{steps}\n\nReferencia técnica: `{self.technical_type}` (`{self.code}`)."


def explain_missing_credentials(names: list[str]) -> UserFacingError:
    readable = ", ".join(f"`{name}`" for name in names)
    return UserFacingError(
        "MISSING_CREDENTIALS_PREFLIGHT",
        "No puedo iniciar la descarga porque faltan credenciales",
        f"La comprobación previa ha detectado que faltan: {readable}. No se ha iniciado el notebook ni se ha realizado ninguna descarga.",
        ("Abre «Credenciales para esta sesión» en la barra lateral.", "Completa los campos indicados y pulsa «Guardar en esta sesión».", "Comprueba que aparecen con estado configurado y vuelve a generar el dataset."),
        "CredentialPreflight",
    )


def explain_exception(error: BaseException, *, stage: str = "pipeline") -> UserFacingError:
    """Traduce errores conocidos sin enviar el traceback ni posibles secretos al chat."""
    text = f"{type(error).__name__}: {error}".lower()
    technical_type = type(error).__name__

    rules: list[tuple[tuple[str, ...], UserFacingError]] = [
        (("completa sh_client_id", "sh_client_id", "sh_client_secret", "oauth de sentinel hub"), UserFacingError(
            "MISSING_SENTINEL_HUB_CREDENTIALS", "Faltan credenciales OAuth de Sentinel Hub",
            "El recorte remoto de Sentinel-3 necesita un Client ID y un Client Secret creados en Sentinel Hub.",
            ("Abre «Credenciales para esta sesión» en la barra lateral.", "Introduce SH_CLIENT_ID y SH_CLIENT_SECRET.", "Guarda y vuelve a pulsar «Generar dataset»."), technical_type)),
        (("completa cds_api_key", "cds_api_key"), UserFacingError(
            "MISSING_CDS_CREDENTIAL", "Falta el token de Climate Data Store",
            "Earth Engine no pudo completar ERA5-Land y el fallback CDS no tiene un token configurado.",
            ("Revisa primero el acceso al proyecto de Earth Engine.", "Si quieres usar el fallback, introduce CDS_API_KEY en la barra lateral.", "Reintenta la ejecución."), technical_type)),
        (("401", "unauthorized", "invalid credential", "invalid_grant"), UserFacingError(
            "AUTHENTICATION_REJECTED", "El servicio ha rechazado las credenciales",
            "Las credenciales sí llegaron al servicio, pero este no las considera válidas o han caducado.",
            ("Comprueba usuario, contraseña o token.", "Si las credenciales estuvieron expuestas, genera unas nuevas.", "Actualízalas en el formulario y reintenta."), technical_type)),
        (("403", "forbidden", "permission denied", "not authorized"), UserFacingError(
            "PERMISSION_DENIED", "La cuenta no tiene permiso para este recurso",
            "La autenticación parece existir, pero falta autorización para el proyecto o dataset solicitado.",
            ("Revisa el proyecto de Earth Engine y los permisos de la cuenta.", "Acepta las condiciones del dataset en su portal si es necesario.", "Reintenta cuando el acceso esté activo."), technical_type)),
        (("429", "too many requests", "rate limit", "quota exceeded"), UserFacingError(
            "RATE_LIMIT", "El servicio ha limitado temporalmente las peticiones",
            "Se ha alcanzado una cuota o se han realizado demasiadas consultas en poco tiempo.",
            ("Espera unos minutos.", "Mantén OVERWRITE_OUTPUTS desactivado para reutilizar la caché.", "Reintenta con un periodo o ROI menor si persiste."), technical_type)),
        (("timeout", "timed out", "connectionerror", "name resolution", "dns"), UserFacingError(
            "NETWORK_ERROR", "No se pudo completar la conexión con un servicio externo",
            "La petición agotó el tiempo de espera o no pudo resolver/conectar con el servidor.",
            ("Comprueba la conexión a Internet y VPN/proxy.", "Verifica que el portal del proveedor esté disponible.", "Reintenta; las cachés existentes se conservarán."), technical_type)),
        (("no scenes", "no products", "no acquisitions", "no se encontraron", "sin adquisiciones"), UserFacingError(
            "NO_PRODUCTS", "No se encontraron productos para los filtros actuales",
            "El área, las fechas, la nubosidad o la franja horaria no han producido adquisiciones utilizables.",
            ("Amplía el intervalo temporal.", "Revisa el ROI.", "Relaja de forma consciente el umbral de nubosidad y vuelve a validar."), technical_type)),
        (("no se encontró ningún lugar", "nominatim", "geocod"), UserFacingError(
            "GEOCODING_ERROR", "No se pudo localizar el lugar descrito",
            "El servicio cartográfico no encontró un resultado inequívoco o no estaba disponible.",
            ("Añade municipio y provincia al nombre.", "Prueba con un barrio, calle o punto de referencia más concreto.", "Comprueba Internet y vuelve a intentarlo."), technical_type)),
        (("modulenotfounderror", "no module named", "importerror"), UserFacingError(
            "MISSING_DEPENDENCY", "Falta una dependencia en el entorno de Python",
            "El kernel seleccionado no contiene una de las librerías necesarias.",
            ("Selecciona el venv del proyecto en VS Code.", "Ejecuta pip install -r requirements-agent.txt.", "Reinicia Streamlit y vuelve a probar."), technical_type)),
        (("kernel died", "deadkernelerror", "kernel is dead"), UserFacingError(
            "KERNEL_DIED", "El kernel del notebook se ha detenido",
            "El proceso de Python terminó durante la ejecución, normalmente por memoria o por un fallo nativo.",
            ("Cierra otros procesos que consuman memoria.", "Prueba con un ROI y periodo menores.", "Reinicia Streamlit y revisa el último bloque ejecutado."), technical_type)),
        (("memoryerror", "out of memory", "cannot allocate memory"), UserFacingError(
            "OUT_OF_MEMORY", "No hay memoria suficiente para esta ejecución",
            "La combinación de ROI, fechas o productos supera la memoria disponible.",
            ("Reduce el ROI o el periodo.", "Divide el intervalo en ejecuciones más pequeñas.", "Cierra otras aplicaciones y reintenta."), technical_type)),
        (("no space left", "disk full"), UserFacingError(
            "DISK_FULL", "No queda espacio suficiente en disco",
            "No se pueden guardar los productos descargados o los CSV de salida.",
            ("Libera espacio en la unidad del proyecto.", "Revisa runs/ y data_downloads/cache.", "No borres cachés útiles hasta comprobar qué ejecución las usa."), technical_type)),
        (("filenotfounderror", "no such file or directory"), UserFacingError(
            "FILE_NOT_FOUND", "Falta un archivo esperado por el pipeline",
            "El notebook no encuentra una entrada, producto descargado o salida intermedia.",
            ("Comprueba que ejecutas Streamlit desde la raíz del proyecto.", "Revisa que nodos_originales_100m.csv exista.", "Consulta el notebook parametrizado de la ejecución para identificar la ruta."), technical_type)),
        (("parsererror", "emptydataerror", "ninguna columna solicitada"), UserFacingError(
            "INVALID_OUTPUT", "La salida generada está vacía o no contiene las columnas solicitadas",
            "El pipeline terminó parcialmente, pero no produjo datos compatibles con la petición.",
            ("Revisa los filtros de calidad y las adquisiciones encontradas.", "Comprueba el CSV original de Landsat o Sentinel-3.", "Ajusta la petición y reintenta."), technical_type)),
    ]

    for needles, result in rules:
        if any(needle in text for needle in needles):
            return result

    title = "No se pudo interpretar la petición" if stage == "interpretation" else "La ejecución no pudo completarse"
    return UserFacingError(
        "UNCLASSIFIED_ERROR", title,
        "El error todavía no pertenece al catálogo de recuperación. Se ha conservado su tipo técnico sin mostrar el traceback ni credenciales en el chat.",
        ("Revisa la última celda del notebook parametrizado.", "Conserva el mensaje técnico para añadir una nueva regla al catálogo.", "No repitas una descarga grande hasta identificar la causa."),
        technical_type,
    )
