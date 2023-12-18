# generador_texto.py
import openai
import pygame
import os
#Ingresar dentro de las "" su key correspondiente de la API a utilizar.
#openai.api_key = ""

num_audio = 0

def generar_frase(frase: list, modelo="text-davinci-003", max_tokens=2048):
    frase_cadena = " ".join(frase)

    prompt = f"Convierte la siguiente frase en una frase con lenguaje fluido, con conectores y que tenga coherencia: {frase_cadena}"
    prompt2 = f"Añade a la siguiente frase conectores, y conjuga sus verbos para lograr que la frase tenga coherencia y cohesión: {frase_cadena}"
    prompt3 = f"{frase_cadena}. A la frase anterior, no añadas ninguna palabra, excepto conectores, y conjuga sus verbos para lograr que la frase tenga cohesión."
    prompt4 = f"{frase_cadena}. To the previous sentence, do not add any words, except connectors, and conjugate its verbs to make the sentence cohesive. You can replace words if necessary."
    # Prompt 5 en español: La frase anterior es una traducción literal de lengua de señas, y eres un experto en interpretación de lengua de señas. Tienes que darle cohesión a la frase. No añadas ninguna palabra, excepto conectores, y conjuga sus verbos adecuadamente para que la persona sorda se dé a entender. Puedes reemplazar palabras si es necesario. Si hay letras separadas y en mayúsculas, júntalas para crear la palabra correcta mas parecida, generalmente son sustantivos propios.
    prompt5 = f"{frase_cadena}. The previous sentence is a literal sign language translation, and you are an expert in sign language interpretation. You have to give cohesion to the sentence. Do not add any words, except connectors, and conjugate your verbs properly so that the deaf person is understood. You can replace words if necessary. If there are separate capitalized letters, put them together to create the closest correct word, usually proper nouns."
    prompt6 = f"{frase_cadena}. La frase anterior es una traducción literal de lengua de señas, y eres un experto en interpretación de lengua de señas. Tienes que darle cohesión a la frase. No añadas ninguna palabra, excepto conectores, y conjuga sus verbos adecuadamente para que la persona sorda se dé a entender. Puedes reemplazar palabras si es necesario. Si hay letras separadas y en mayúsculas, júntalas para crear la palabra correcta mas parecida, y deja la palabra en minúscula."
    prompt7 = f"Genera una respuesta natural y coherente a partir de la siguiente entrada, corrigiendo la capitalización y la estructura gramatical según las convenciones del lenguaje humano: {frase_cadena}"
    prompt8 = f"Utilizando las palabras proporcionadas, corrige la capitalización y la estructura gramatical para formar una respuesta coherente: {frase_cadena}"
    
    # Tokens aproximados
    # prompt4:  37 tokens
    # prompt6:  85 tokens

    respuesta = openai.completions.create(
        model=modelo,
        prompt=prompt8,
        max_tokens=max_tokens
    )

    completion = respuesta.choices[0].text
    
    return completion

def generar_tts(cadena: str, voz="alloy"):
    global num_audio
    ruta_audio = f"audio/frase{num_audio}.mp3"
    
    respuesta = openai.audio.speech.create(
        model="tts-1",
        voice=voz,
        input=cadena
    )

    respuesta.stream_to_file(ruta_audio)
    
    pygame.mixer.init()
    pygame.mixer.music.load(ruta_audio)
    pygame.mixer.music.play(loops=0)
    
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    
    if num_audio > 0:
        os.remove(f"audio/frase{num_audio-1}.mp3")
    
    num_audio += 1
    
def borrar_audios(carpeta="audio"):
    for nombre_archivo in os.listdir(carpeta):
        if nombre_archivo.endswith('.mp3'):
            try:
                os.remove(os.path.join(carpeta, nombre_archivo))
            except:
                pass