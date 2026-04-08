import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader
from dotenv import load_dotenv

# 1. Configuración inicial
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configurar el motor Groq (Súper rápido y sin bloqueos)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ Error: Falta la variable GROQ_API_KEY en el archivo .env")

client = Groq(api_key=groq_api_key)

# 3. Leer el PDF directamente a la memoria y ajustarlo al límite de Groq
def extraer_texto_pdf(ruta_pdf):
    texto = ""
    try:
        reader = PdfReader(ruta_pdf)
        for page in reader.pages:
            texto += page.extract_text() + "\n"
        
        # --- EL TRUCO DE COMPRESIÓN ---
        # 1. Quitamos espacios y saltos de línea extra que consumen memoria
        texto_limpio = " ".join(texto.split())
        
        # 2. Cortamos el texto a un tamaño 100% seguro para Groq (aprox 4500 tokens)
        if len(texto_limpio) > 18000:
            texto_limpio = texto_limpio[:18000]
            
        print("📄 PDF cargado y comprimido exitosamente para Groq.")
        return texto_limpio
    except Exception as e:
        print(f"❌ Error leyendo el PDF: {e}")
        return ""

print("🚀 Iniciando UniBot con motor Groq...")
texto_documento = extraer_texto_pdf("faq.pdf")

class UserQuery(BaseModel):
    pregunta: str
    session_id: str = "anonimo"

@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    print(f"📩 Pregunta recibida: {query.pregunta}")

    # Saludo inicial fijo
    saludos = ["hola", "buen dia", "buenas", "que tal"]
    if query.pregunta.lower().strip() in saludos:
        return {"respuesta": "¡Hola! 👋 Soy UniBot, el asistente de la UNCAus. ¿En qué te puedo ayudar hoy?"}

    # --- EL NUEVO PROMPT: CORTO, CONCISO Y DIRECTO ---
    prompt = f"""
    Eres UniBot, el asistente virtual oficial de la Universidad Nacional del Chaco Austral (UNCAus).
    
    Documento oficial con la información:
    ---------------------
    {texto_documento}
    ---------------------
    
    Pregunta del estudiante: {query.pregunta}
    
    Reglas IMPORTANTES para tu respuesta:
    1. Directo al grano: Responde de manera MUY CORTA y CONCISA. Ve directo a la solución sin introducciones largas ni rodeos.
    2. Sin saludos repetitivos: NO empieces tu respuesta diciendo "Hola" ni saludando. Entregá la información directamente.
    3. Formato visual: Usa viñetas (-) y resalta los requisitos o palabras importantes en **negrita** para que se lea rápido. (Podés usar algún emoji como ✅ o 📌).
    4. Cierre: NO hagas preguntas de cierre como "¿Te puedo ayudar en algo más?". Termina la respuesta y listo.
    5. Precisión: Basa tu respuesta SOLO en el documento. Si no está ahí, responde brevemente: "No tengo ese dato oficial. Por favor, consultá con la sección de Alumnado."
    """ 

    try:
        # Usamos el modelo correcto y actual de Llama 3.1 en Groq
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", 
        )
        return {"respuesta": chat_completion.choices[0].message.content}
    except Exception as e:
        print(f"❌ Error de Groq: {e}")
        return {"respuesta": "Lo siento, tengo un problema técnico en este momento. Intenta de nuevo en unos minutos."}

@app.get("/")
def home():
    return {"status": "UniBot v13.0 - VERSIÓN FINAL CONCISA ⚡"}