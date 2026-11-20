import os
import secrets
import io
import time
import base64
import logging
import json
import asyncio
import httpx
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from PIL import Image
from fastapi import FastAPI, HTTPException, Security, Depends, status, Request, BackgroundTasks, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


# ------------------------------------------------------------------
# ENTORNO Y CONFIGURACIÓN
# ------------------------------------------------------------------
load_dotenv()

MODELS_FALLBACK = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"]
NODE_CALLBACK_URL = os.getenv("NODE_CALLBACK_URL")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING ESTRUCTURADO
# ------------------------------------------------------------------
class DefaultTagFilter(logging.Filter):
    """Inyecta un tag 'SYSTEM' por defecto para logs de librerías externas."""
    def filter(self, record):
        if not hasattr(record, "tag"):
            record.tag = "SYSTEM"
        return True

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(tag)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

for handler in logging.getLogger().handlers:
    handler.addFilter(DefaultTagFilter())

class Logger:
    @staticmethod
    def info(tag: str, msg: str, data: Any = None):
        extra = f"\n{json.dumps(data, indent=2, ensure_ascii=False)}" if isinstance(data, (dict, list)) else (f"\n{data}" if data else "")
        logging.info(f"{msg}{extra}", extra={"tag": tag})

    @staticmethod
    def success(tag: str, msg: str, data: Any = None):
        extra = f"\n{json.dumps(data, indent=2, ensure_ascii=False)}" if isinstance(data, (dict, list)) else (f"\n{data}" if data else "")
        logging.info(f"✅ {msg}{extra}", extra={"tag": tag})

    @staticmethod
    def warn(tag: str, msg: str, data: Any = None):
        extra = f"\n{json.dumps(data, indent=2, ensure_ascii=False)}" if isinstance(data, (dict, list)) else (f"\n{data}" if data else "")
        logging.warning(f"⚠️ {msg}{extra}", extra={"tag": tag})

    @staticmethod
    def error(tag: str, msg: str, data: Any = None):
        extra = f"\n{json.dumps(data, indent=2, ensure_ascii=False)}" if isinstance(data, (dict, list)) else (f"\n{data}" if data else "")
        logging.error(f"❌ {msg}{extra}", extra={"tag": tag})

log = Logger()

def mask_phone(phone: Optional[str]) -> str:
    """Enmascara un teléfono en logs, dejando solo los últimos 4 dígitos."""
    if not phone:
        return "?"
    s = str(phone)
    return f"***{s[-4:]}" if len(s) > 4 else "***"

log.info("STARTUP", "Iniciando AI Agent Microservice (Gemini)...")
log.info("ENV_CHECK", "Estado de variables de entorno:", {
    "INTERNAL_API_KEY": "✅ Configurada" if INTERNAL_API_KEY else "❌ FALTANTE",
    "GEMINI_API_KEY": "✅ Configurada" if GEMINI_API_KEY else "❌ FALTANTE",
    "NODE_CALLBACK_URL": "✅ Configurada" if NODE_CALLBACK_URL else "❌ FALTANTE"
})

if not GEMINI_API_KEY:
    log.error("GEMINI", "Error crítico: GEMINI_API_KEY no se encuentra definida.")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="AI Agentic Microservice (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://slateblue-horse-642737.hostingersite.com"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# ------------------------------------------------------------------
# MIDDLEWARE Y SEGURIDAD UNIFICADA
# ------------------------------------------------------------------
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_internal_key(api_key: str = Security(api_key_header)):
    if not INTERNAL_API_KEY or not api_key:
        log.error("AUTH_FAIL", "Intento de acceso rechazado: X-API-Key ausente.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Acceso no autorizado: X-API-Key ausente"
        )
    
    # Comparación en tiempo constante (protección contra timing attacks)
    if not secrets.compare_digest(api_key, INTERNAL_API_KEY):
        log.error("AUTH_FAIL", "Intento de acceso rechazado: X-API-Key inválida. ")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Acceso no autorizado: X-API-Key inválida"
        )
        
    log.success("AUTH_OK", "Autenticación interna validada correctamente.")
    return api_key

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_host = request.client.host if request.client else "Unknown"
    log.info("HTTP_IN", f"--> {request.method} {request.url.path} [IP: {client_host}]")
    
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    
    log.info("HTTP_OUT", f"<-- {request.method} {request.url.path} {response.status_code} [{duration}ms]")
    return response

# ------------------------------------------------------------------
# ESQUEMAS PYDANTIC (Structured Outputs)
# ------------------------------------------------------------------
class RegionViaje(str, Enum):
    # Las 6 regiones geográficas oficiales de Argentina.
    NOROESTE_ARG = "Noroeste (NOA)"
    NORESTE_ARG = "Noreste (NEA)"
    CUYO = "Cuyo"
    PAMPEANA = "Pampeana"
    PATAGONIA_ARG = "Patagonia"
    CABA = "CABA"
    # Internacional, por continente/subregión.
    CARIBE = "Caribe"
    CENTROAMERICA = "Centroamérica"
    SUDAMERICA_RESTO = "Sudamérica (resto)"
    NORTEAMERICA = "Norteamérica"
    EUROPA = "Europa"
    ASIA = "Asia"
    OCEANIA = "Oceanía"
    MEDIO_ORIENTE = "Medio Oriente"
    AFRICA = "África"
    OTRO = "Otro / No identificado"

class OfertaViaje(BaseModel):
    destino: str = Field(description="Ciudad o región principal del viaje (ej: 'Bariloche', 'Mar del Plata')")
    region: RegionViaje = Field(description=(
        "Zona geográfica del destino, para poder sugerir alternativas "
        "cercanas cuando no hay stock de lo que se pide. Si el destino es "
        "argentino, usá la región geográfica oficial que le corresponde: "
        "'Noroeste (NOA)' (Jujuy, Salta, Tucumán, Catamarca, La Rioja, "
        "Santiago del Estero), 'Noreste (NEA)' (Misiones, Corrientes, "
        "Chaco, Formosa -- incluye Iguazú), 'Cuyo' (Mendoza, San Juan, San "
        "Luis), 'Pampeana' (provincia de Buenos Aires incluida su costa "
        "atlántica como Mar del Plata/Pinamar, Córdoba, Santa Fe, Entre "
        "Ríos, La Pampa), 'Patagonia' (Neuquén, Río Negro, Chubut, Santa "
        "Cruz, Tierra del Fuego -- incluye Bariloche, Ushuaia, El "
        "Calafate), o 'CABA' SOLO para la Ciudad Autónoma de Buenos Aires "
        "(no la provincia). Si es un destino internacional, usá el "
        "continente o subregión que corresponda (ej: Punta Cana/Cancún -> "
        "'Caribe'; Brasil -> 'Sudamérica (resto)'; España/Italia -> "
        "'Europa'). Usá 'Otro / No identificado' SOLO si el destino no "
        "encaja claramente en ninguna de estas categorías."
    ))
    fecha_salida: str = Field(description=(
        "UNA sola fecha de salida, escrita SIEMPRE con este formato: "
        "'<día> de <mes>' con el mes completo en minúsculas (ej: '3 de octubre', "
        "'15 de noviembre'), y ' de <año>' al final SOLO si el año figura en el "
        "afiche (ej: '15 de noviembre de 2026'). Convertí abreviaturas y números "
        "('15-nov', '15/11', 'Sáb 15 Nov') a ese formato, sin el día de la semana. "
        "NO inventes años. Si no hay un día concreto, copiá el texto tal cual "
        "(ej: 'A confirmar', 'Salidas los sábados de enero'). Si el afiche lista "
        "varias fechas para este mismo destino, esta oferta representa SOLO una de "
        "ellas -- nunca una lista ni un rango de fechas acá."
    ))
    duracion: Optional[str] = Field(None, description="Duración del viaje (ej: '6 Días / 3 Noches')")
    duracion_dias: Optional[int] = Field(None, description=(
        "Cantidad de DÍAS del viaje como número, tal como figura en el afiche "
        "(ej: '6 Días / 3 Noches' -> 6). null si el afiche no dice los días."
    ))
    duracion_noches: Optional[int] = Field(None, description=(
        "Cantidad de NOCHES (de alojamiento) como número, tal como figura en el afiche "
        "(ej: '6 Días / 3 Noches' -> 3; '7 noches' -> 7). null si el afiche no dice las "
        "noches. No la calcules a partir de los días."
    ))
    hotel: Optional[str] = Field(None, description="Nombre del hotel o hospedaje (ej: 'Hotel Quime Quipan Inn')")
    regimen_comida: Optional[str] = Field(None, description="Tipo de comida (ej: 'Desayuno', 'Media Pensión', 'Pensión Completa')")
    inclusiones: Optional[str] = Field(None, description="Servicios adicionales incluidos (ej: 'Traslado, Alojamiento, Coordinador')")
    precio_promo: Optional[float] = Field(None, description="Precio total final o promocional publicado (ej: 249990)")
    promocion: Optional[str] = Field(None, description="Texto o regla de la promoción si figura (ej: 'Viajan 3 Pagan 2')")
    contacto: Optional[str] = Field(None, description="Teléfono o medio de contacto extraído del afiche")
    cupos: Optional[int] = Field(1, description="Cantidad estimada de cupos o plazas si se mencionan")

class ListaOfertasViaje(BaseModel):
    ofertas: List[OfertaViaje] = Field(description="Lista de todas las ofertas individuales detectadas")

class FlyerPayload(BaseModel):
    phone_number: str
    text_content: Optional[str] = ""
    image_base64: Optional[str] = None
    mime_type: Optional[str] = "image/jpeg"
    # Id de la fila en 'fuentes_extraccion' (Node ya la creó antes de llamar
    # acá). Python no la usa para nada, solo la devuelve en el callback para
    # que Node pueda vincular cada oferta encontrada a su fuente original.
    fuente_id: Optional[int] = None

class ChatMessage(BaseModel):
    role: str  # "user" o "model"
    content: str

class ChatRequest(BaseModel):
    user_message: str
    travel_catalog: List[dict]
    busqueda_sin_resultados: Optional[dict] = None
    history: Optional[List[ChatMessage]] = []

class ConfirmAction(str, Enum):
    APROBAR = "APROBAR"
    RECHAZAR = "RECHAZAR"
    DESCONOCIDO = "DESCONOCIDO"

class ConfirmResult(BaseModel):
    accion: ConfirmAction
    notas: Optional[str] = None

class ConfirmRequest(BaseModel):
    admin_message: str

class IntentPayload(BaseModel):
    user_message: str
    destinos_conocidos: List[str] = []

class IntentExtraido(BaseModel):
    destino: Optional[str] = Field(None, description="El lugar/destino que pide el cliente. Si coincide con uno de destinos_conocidos, escribilo EXACTAMENTE igual que en esa lista (para poder filtrar el catálogo). Si el cliente pide un lugar real y concreto que NO está en destinos_conocidos (ej. pregunta por una ciudad/provincia/país que la agencia no tiene activa ahora), escribí igual el nombre de ese lugar tal como lo entendiste -- NO lo pongas en None solo porque no está en la lista, hace falta igual para poder sugerirle algo cercano. Usá None únicamente si el cliente no menciona ningún lugar geográfico concreto (ej. 'hola', '¿qué tienen?').")
    mes: Optional[int] = Field(None, description="Mes de salida que pide, del 1 al 12, si lo menciona explícita o relativamente (ej: 'el mes que viene', 'para fin de año', resolviendo sobre la fecha de hoy que se da en el prompt). None si no lo menciona.")
    busca_promocion: bool = Field(False, description="True SOLO si el cliente pregunta específicamente por promociones, descuentos u ofertas especiales.")
    consulta_especifica: bool = Field(..., description="True si el mensaje pregunta por algo concreto (destino, fecha, tipo de viaje, promoción). False si es un saludo genérico o una pregunta vaga sin ningún criterio identificable (ej: 'hola', '¿qué tienen?', 'quiero viajar').")

# ------------------------------------------------------------------
# FUNCIONES AUXILIARES Y GEMINI FALLBACK
# ------------------------------------------------------------------
def optimizar_flyer_para_vision(imagen_bytes: bytes, max_size: int = 1024) -> Image.Image:
    img = Image.open(io.BytesIO(imagen_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

# Tiempo máximo que puede esperar cada llamado. Vercel corta la función a los
# `maxDuration` segundos (vercel.json) y, si eso pasa, la tarea muere sin
# avisarle nada a Node. Por eso cada intento lleva su propio timeout y nunca
# se pasa del plazo total (`deadline`): así siempre queda tiempo para mandar el
# aviso de error. Sin deadline (chat, intención, confirmación: Node espera la
# respuesta hasta 60 s) se usan 50 s.
PLAZO_SINCRONICO_S = 50
POR_INTENTO_SINCRONICO_S = 25   # chat: un modelo colgado no se come los 50 s
POR_INTENTO_FLYER_S = 90        # afiche: imagen + JSON largo, tarda más
MINIMO_PARA_INTENTAR_S = 12  # Gemini rechaza timeouts de menos de 10 s

async def generate_with_fallback(contents, system_instruction, response_schema=None, deadline=None, por_intento=None):
    last_error = None
    if deadline is None:
        deadline = time.time() + PLAZO_SINCRONICO_S
    if por_intento is None:
        por_intento = POR_INTENTO_SINCRONICO_S

    for model_name in MODELS_FALLBACK:
        for attempt in range(2):
            restante = deadline - time.time()
            if restante < MINIMO_PARA_INTENTAR_S:
                log.warn("GEMINI_SIN_TIEMPO", f"Quedan {restante:.0f}s: no se prueba {model_name}.")
                ultimo = f" Último error: {str(last_error)[:200]}" if last_error else ""
                raise Exception(f"Gemini tardó demasiado en responder (servicio saturado). Probá de nuevo en unos minutos.{ultimo}")
            try:
                log.info("GEMINI_CALL", f"Invocando {model_name} (Intento {attempt + 1}, hasta {min(restante, por_intento):.0f}s)...")
                start_time = time.time()

                config_args = {
                    "system_instruction": system_instruction,
                    "temperature": 0.2,
                    # en milisegundos: si el modelo no contesta, se pasa al siguiente
                    "http_options": types.HttpOptions(timeout=int(max(10, min(restante, por_intento)) * 1000))
                }
                if response_schema:
                    config_args["response_mime_type"] = "application/json"
                    config_args["response_schema"] = response_schema

                # Ejecutar la llamada sincrónica en un hilo asíncrono para no bloquear el Event Loop
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_args)
                )
                
                duration = round((time.time() - start_time) * 1000, 2)
                log.success("GEMINI_RES", f"Respuesta obtenida con {model_name} en {duration}ms")
                return response

            except Exception as e:
                last_error = e
                err_str = str(e)
                log.warn("GEMINI_RETRY", f"Falla con {model_name}: {err_str[:120]}... Reintentando.")
                
                # 503/429: saturado un momento, se reintenta una vez el mismo
                # modelo. 504, timeout u otro error: directo al siguiente modelo.
                if ("503" in err_str or "429" in err_str) and attempt == 0:
                    await asyncio.sleep(1.5)
                else:
                    break
                    
    # "Saturados" solo si el último intento fue justamente eso (503/429) --
    # si fue un 404 (modelo retirado/renombrado por Google) u otro error, el
    # mensaje anterior confundía el diagnóstico (parecía un problema de
    # carga cuando en realidad era un modelo que ya no existe).
    ultimo_str = str(last_error)
    motivo = "Servicios de Google saturados" if ("503" in ultimo_str or "429" in ultimo_str) else "No se pudo completar con ningún modelo de la lista de fallback"
    raise Exception(f"{motivo}. Último error: {ultimo_str}")

# ------------------------------------------------------------------
# TAREA DE FONDO (ASYNC CALLBACK)
# ------------------------------------------------------------------
# Plazo total del afiche: el maxDuration de vercel.json menos un margen para
# el callback a Node (éxito o error). Configurable con FLYER_PLAZO_S.
FLYER_PLAZO_S = float(os.getenv("FLYER_PLAZO_S", "270"))

async def task_extract_flyer_and_notify(payload: FlyerPayload):
    deadline = time.time() + FLYER_PLAZO_S
    try:
        log.info("GEMINI_TASK", f"Iniciando procesamiento de afiche para {mask_phone(payload.phone_number)}...")
        contents = []

        if payload.image_base64:
            try:
                image_bytes = base64.b64decode(payload.image_base64)
                img_pil = optimizar_flyer_para_vision(image_bytes)
                contents.append(img_pil)
            except Exception as e:
                log.error("IMAGE_PROC_ERR", f"Error procesando imagen Base64: {str(e)}")

        if payload.text_content:
            contents.append(f"Texto del mensaje/flyer:\n{payload.text_content}")

        if not contents:
            raise Exception("No se proporcionó información ni imagen para procesar.")

        system_instruction = (
            "Eres un extractor estricto de datos de afiches publicitarios de viajes.\n"
            "REGLAS OBLIGATORIAS:\n"
            "1. Extrae los datos LITERALMENTE como figuran en la imagen o texto.\n"
            "2. NO inventes un año en la fecha de salida si no figura explícitamente en el afiche. "
            "Escribí la fecha como '<día> de <mes>' (ej: '15 de noviembre'), con ' de <año>' "
            "solo si el año está en el afiche (ver el campo fecha_salida).\n"
            "3. NO realices cálculos matemáticos ni divisiones de precios por persona por tu cuenta.\n"
            "4. Si un afiche contiene múltiples destinos, extráelos como ofertas separadas.\n"
            "5. Si para UN MISMO destino hay varias fechas de salida -- por ejemplo "
            "'Salidas: 14, 17, 23, 26 y 29 de Noviembre' o 'Sale los martes y jueves' -- "
            "generá UNA oferta separada POR CADA FECHA individual, repitiendo igual el "
            "resto de los datos (hotel, precio, régimen, cupos, etc.) en cada una. "
            "NUNCA pongas una lista ni un rango de fechas en fecha_salida: cada oferta "
            "lleva una sola fecha.\n"
            "6. Clasificá region según la definición del campo -- es para poder sugerir "
            "destinos cercanos cuando no haya stock exacto de lo pedido, así que tiene "
            "que ser consistente entre afiches (el mismo destino siempre a la misma "
            "región)."
        )

        response = await generate_with_fallback(
            contents=contents,
            system_instruction=system_instruction,
            response_schema=ListaOfertasViaje,
            deadline=deadline,
            por_intento=POR_INTENTO_FLYER_S
        )

        datos_dict = json.loads(response.text)
        log.success("GEMINI_DONE", f"Extracción completada con éxito para {mask_phone(payload.phone_number)}")

        # Enviar el resultado de vuelta a Node.js vía Webhook Callback
        if NODE_CALLBACK_URL:
            async with httpx.AsyncClient(timeout=30.0) as client_http:
                res = await client_http.post(
                    NODE_CALLBACK_URL,
                    json={
                        "phoneNumber": payload.phone_number,
                        "extractedData": datos_dict,
                        "fuenteId": payload.fuente_id
                    },
                    headers={
                        "X-API-Key": INTERNAL_API_KEY,
                        "Authorization": f"Bearer {INTERNAL_API_KEY}",
                        "Content-Type": "application/json"
                    }
                )
                log.info("CALLBACK_SENT", f"Resultado enviado a Node.js (Status: {res.status_code})")

    except Exception as e:
        log.error("TASK_ERR", f"Error en la tarea de fondo de Gemini: {type(e).__name__}: {e!r} (callback: {NODE_CALLBACK_URL})")
        if NODE_CALLBACK_URL:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client_http:
                    await client_http.post(
                        NODE_CALLBACK_URL,
                        json={
                            "phoneNumber": payload.phone_number,
                            "error": True,
                            "message": f"Error procesando la imagen: {str(e)}"
                        },
                        headers={
                            "X-API-Key": INTERNAL_API_KEY,
                            "Authorization": f"Bearer {INTERNAL_API_KEY}"
                        }
                    )
            except Exception as notify_err:
                log.error("CALLBACK_FAIL", f"No se pudo notificar el error a Node.js: {type(notify_err).__name__}: {notify_err!r} (callback: {NODE_CALLBACK_URL})")

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "OK", "service": "Python AI Microservice (Gemini)"}

@app.post("/agent/extract-flyer", dependencies=[Depends(verify_internal_key)])
async def extract_flyer(payload: FlyerPayload, background_tasks: BackgroundTasks):
    log.info("FLYER_REQ", f"Petición recibida para {mask_phone(payload.phone_number)}. Programando tarea en segundo plano...")
    
    background_tasks.add_task(task_extract_flyer_and_notify, payload)
    
    return {
        "ok": True,
        "status": "processing",
        "message": "Afiche recibido correctamente. Procesando en segundo plano."
    }

@app.post("/agent/chat", dependencies=[Depends(verify_internal_key)])
async def chat_agent(payload: ChatRequest):
    # No se loguea el contenido del mensaje del cliente, solo su longitud.
    log.info("CHAT_REQ", f"Consulta recibida ({len(payload.user_message)} caracteres)")

    try:
        instruccion_sin_match = ""
        if payload.busqueda_sin_resultados:
            busqueda = payload.busqueda_sin_resultados
            region_sugerida = busqueda.get("regionSugerida") if isinstance(busqueda, dict) else None
            pista_region = (
                f" El catálogo de arriba ya viene limitado a la región "
                f"'{region_sugerida}' (la misma del destino que pidió), así que "
                f"elegí la alternativa directamente de ahí."
                if region_sugerida else
                " No hay otro destino en la misma región, así que el catálogo de "
                "arriba es el completo -- elegí la alternativa que tenga la fecha "
                "de salida más próxima a la que pidió, de cualquier destino."
            )
            instruccion_sin_match = f"""

        ATENCIÓN -- caso especial de esta consulta puntual: el cliente pidió
        algo específico ({busqueda}) y NO hay ninguna salida activa que
        coincida con eso exactamente.{pista_region}
        Tu tarea en este mensaje es:
        a) Decirle brevemente que no hay disponibilidad exacta para lo que
           pidió.
        b) Sugerirle proactivamente 1-3 alternativas puntuales que SÍ estén
           en el catálogo de arriba -- nunca algo que no figure literalmente
           ahí.
        c) No enumeres el catálogo completo, mencioná solo esas alternativas,
           como haría un vendedor atento.
        """

        system_instruction = f"""
        Eres un asesor de viajes por WhatsApp atento y comercial.

        Tenés acceso a este catálogo completo de viajes disponibles con stock
        (es tu referencia interna, NO se lo muestres ni lo resumas completo
        al cliente de entrada):
        {payload.travel_catalog}

        Tus reglas:
        1. Respondé ÚNICAMENTE basándote en este catálogo.
        2. NO enumeres ni resumas el catálogo completo salvo que el cliente
           lo pida explícitamente. Si el mensaje es genérico (ej: "hola",
           "¿qué viajes tienen?", sin especificar destino/fecha/tipo de
           viaje), NO listes opciones: preguntale primero qué destino, fecha
           o tipo de viaje le interesa, para poder responder algo puntual en
           vez de mandarle todo el catálogo de una.
        3. Cuando el cliente sí pregunte por algo puntual, mencioná
           claramente el destino, fecha, y si quedan POCOS cupos (mencioná
           los cupos si son < 3). Si hay varias opciones que coinciden con
           lo que pidió, está bien mostrar esas pocas.
        3bis. Cuando el cliente pida "más información" o detalles de una
           opción puntual, SIEMPRE incluí precio (precio_promo), hotel,
           régimen de comida (regimen_comida) e inclusiones si ese dato
           existe en el catálogo -- no te quedes solo con una descripción
           genérica, esos son los datos que un cliente necesita para
           decidir. Si algún campo viene vacío/null, no lo menciones (no
           digas "a consultar" ni inventes un valor).
        4. Si el cliente pregunta por un destino o fecha que NO figura en el
           catálogo, informale amablemente que no hay salidas disponibles
           por el momento para eso, y preguntale qué otro destino o fecha le
           interesaría para poder ofrecerle alternativas reales.
        5. Mantené las respuestas concisas, amables y adecuadas para un chat
           de WhatsApp.
        {instruccion_sin_match}
        """

        contents = []
        for msg in payload.history:
            role = "user" if msg.role.lower() in ["user", "human"] else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
        
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=payload.user_message)]))

        start_gemini = time.time()
        
        # Ejecución asíncrona no bloqueante
        response = await generate_with_fallback(
            contents=contents,
            system_instruction=system_instruction
        )

        duration = round((time.time() - start_gemini) * 1000, 2)
        log.success("CHAT_RES", f"Respuesta generada en {duration}ms ({len(response.text)} caracteres)")

        return {"response": response.text}

    except Exception as e:
        log.error("CHAT_ERR", f"Error en Chat Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en Chat Gemini: {str(e)}")

@app.post("/agent/confirm", response_model=ConfirmResult, dependencies=[Depends(verify_internal_key)])
async def parse_admin_confirmation(payload: ConfirmRequest):
    # No se loguea el contenido del mensaje del admin, solo su longitud.
    log.info("CONFIRM_REQ", f"Evaluando respuesta de admin ({len(payload.admin_message)} caracteres)")

    try:
        prompt = f"Determina si el siguiente mensaje del administrador autoriza la publicación ('APROBAR') o la rechaza ('RECHAZAR'):\n'{payload.admin_message}'"
        
        start_gemini = time.time()
        
        response = await generate_with_fallback(
            contents=prompt,
            system_instruction="Eres un evaluador de intenciones de aprobación.",
            response_schema=ConfirmResult
        )

        duration = round((time.time() - start_gemini) * 1000, 2)
        log.success("CONFIRM_RES", f"Decisión interpretada en {duration}ms")

        return ConfirmResult.model_validate_json(response.text)

    except Exception as e:
        log.error("CONFIRM_ERR", f"Error en Confirmación Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en Confirmación Gemini: {str(e)}")

@app.post("/agent/extract-intent", response_model=IntentExtraido, dependencies=[Depends(verify_internal_key)])
async def extract_intent(payload: IntentPayload):
    # No se loguea el mensaje del cliente, solo su longitud. Llamado liviano
    # (sin catálogo completo) para decidir qué filtrar antes del chat real --
    # ver ahorro de tokens / reducción de falsos negativos en la auditoría.
    log.info("INTENT_REQ", f"Extrayendo intención ({len(payload.user_message)} caracteres, {len(payload.destinos_conocidos)} destinos conocidos)")

    try:
        hoy = datetime.now().strftime("%Y-%m-%d (%A)")
        destinos_texto = ", ".join(payload.destinos_conocidos) if payload.destinos_conocidos else "(no hay destinos con salidas activas en este momento)"

        prompt = (
            f"Hoy es {hoy}. El cliente escribió: \"{payload.user_message}\"\n\n"
            f"Destinos con salidas activas en este momento: {destinos_texto}\n\n"
            "Extraé la intención de este mensaje puntual."
        )

        start_gemini = time.time()

        response = await generate_with_fallback(
            contents=prompt,
            system_instruction=(
                "Extraés la intención de mensajes de clientes de WhatsApp para una "
                "agencia de viajes, para poder buscarles ofertas puntuales en vez de "
                "mostrarles todo el catálogo. Sé conservador: si no estás seguro de "
                "que el destino mencionado coincida con uno de los conocidos, dejalo "
                "en None en vez de adivinar -- es preferible preguntar de nuevo que "
                "decirle al cliente que algo no existe cuando sí existe."
            ),
            response_schema=IntentExtraido
        )

        duration = round((time.time() - start_gemini) * 1000, 2)
        log.success("INTENT_RES", f"Intención extraída en {duration}ms")

        return IntentExtraido.model_validate_json(response.text)

    except Exception as e:
        log.error("INTENT_ERR", f"Error extrayendo intención: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extrayendo intención: {str(e)}")