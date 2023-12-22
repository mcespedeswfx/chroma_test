import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
import os
#----------- ENV VAR -----------------
from dotenv import load_dotenv, find_dotenv

# Cunado se habla de query data es encontrar los chuncks dentro de las bases de datos que respondan la pregunta
# Se necesita la base de datos y los embeddings.
# Se realiza el query y se encuentran los chunks de data que están más cencarnos.
# El AI lee toda la informacion que se paresca, enceuntra los chucks y decide cual es lo que mas se parece

# --------------------------------
#            API KEY 
# --------------------------------
# Se carga la clave de API de OpenAI desde un archivo .env
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
x1 = openai.api_key
embeddings = OpenAIEmbeddings(openai_api_key=x1)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    # Esto es para poder escribir en consola
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    #------------- Prepare the DB.
    # Lo primero que necesito es el path del API para autorizar el uso
    embedding_function = embeddings 


    # Aqui necesito crear un embbeding function que es la misma que utilizo para crear el DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Cuando se carga la base de datos ya se puede buscar por el chuck que mejor match el query
    # Se pasa el query_text com oun argument y se especifica el numero de resultados que deseamos
    # Best matches
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Los resultados vana ser tuplas
    # Antes de analizar los resultados se puede poner filtros
    # Si el resultado no existe o match es menor a 0.7
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    #----------------------------
    #       CREAR PROMPT
    #----------------------------
    # Todo esto lo que hace es juntar todos los chucks de data en una linea y la presenta 
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print(prompt)

    #----------------------------
    #       ESCOJER MODELO
    #----------------------------
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    #----------------------------
    #     REFERENCES
    #----------------------------
    # Provide references back to your original material.
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()