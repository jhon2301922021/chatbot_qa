import streamlit as st
from openai import OpenAI
import pandas as pd
from utils import get_context_from_query, custom_prompt
import json

# Abrir y leer el archivo 'credentials.json' para obtener configuraciones, como la clave API de OpenAI.
file_name = open('credentials.json')
config_env = json.load(file_name)

# Cargar un DataFrame de pandas previamente almacenado en un archivo pickle.
# Este DataFrame contiene embeddings vectoriales y otros datos relevantes.
df_vector_store = pd.read_pickle('df_vector_store.pkl')

# Definir la función principal de la página, que contiene la lógica de la interfaz de usuario y la interacción modelo-usuario.
def main_page():
  
  # Inicializar variables en el estado de la sesión de Streamlit si aún no existen.
  # Esto incluye configuraciones predeterminadas para el modelo y parámetros de generación de texto.
  if "temperature" not in st.session_state:
      st.session_state.temperature = 0.0
  if "model" not in st.session_state:
      st.session_state.model = "gpt-3.5-turbo"
  if "message_history" not in st.session_state:
      st.session_state.message_history = []

  # Usar la barra lateral de Streamlit para mostrar opciones de configuración y recibir entradas del usuario.
  with st.sidebar:
    # Mostrar un logotipo y títulos de secciones utilizando funciones de Streamlit para elementos de UI.
    st.image('dmc_logo.jpg', use_column_width="always")
    st.header(body="Chat personalizado :robot_face:")
    st.subheader('Configuración del modelo :level_slider:')

    # Permitir al usuario elegir entre diferentes modelos de OpenAI.
    model_name = st.radio("**Elije un modelo**:", ("GPT-3.5", "GPT-4"))
    if model_name == "GPT-3.5":
      st.session_state.model = "gpt-3.5-turbo"
    elif model_name == "GPT-4":
      st.session_state.model = "gpt-4"
    
    # Permitir al usuario ajustar el nivel de creatividad (temperatura) de las respuestas generadas.
    st.session_state.temperature = st.slider("**Nivel de creatividad de respuesta**  \n  [Poco creativo ►►► Muy creativo]",
                                             min_value = 0.0,
                                             max_value = 1.0,
                                             step      = 0.1,
                                             value     = 0.0)
    
  # Mostrar el historial de mensajes si ya se ha generado una consulta.
  if st.session_state.get('generar_pressed', False):
    for message in st.session_state.message_history:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

  # Capturar la entrada del usuario a través de un campo de entrada de chat.
  if prompt := st.chat_input("¿Cuál es tu consulta?"):

    # Mostrar la consulta del usuario en el chat.
    with st.chat_message("user"):
        st.markdown(prompt)

    # Preparar y mostrar la respuesta del asistente.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Obtener contextos relevantes del almacenamiento vectorial basado en la consulta.
        Context_List = get_context_from_query(query = prompt,
                                              vector_store = df_vector_store,
                                              n_chunks = 5)
        # Crear una instancia del cliente de OpenAI con la clave API.
        client = OpenAI(api_key=config_env["openai_key"])
        # Generar una respuesta utilizando el modelo y la configuración seleccionados por el usuario.
        completion = client.chat.completions.create(
          model=st.session_state.model,
          temperature = st.session_state.temperature,
          messages=[{"role": "system", "content": f"{custom_prompt.format(source = str(Context_List))}"}] + 
          st.session_state.message_history + 
          [{"role": "user", "content": prompt}]
        )
        # Mostrar la respuesta generada en el chat.
        full_response = completion.choices[0].message.content
        message_placeholder.markdown(full_response)

    # Actualizar el historial de mensajes en el estado de la sesión.
    st.session_state.message_history.append({"role": "user", "content": prompt})
    st.session_state.message_history.append({"role": "assistant", "content": full_response})
    st.session_state.generar_pressed = True

# Punto de entrada de la aplicación Stream
if __name__ == "__main__":
    main_page()