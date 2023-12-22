from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

from pydantic import BaseModel
import openai
#----------- ENV VAR -----------------
from dotenv import load_dotenv, find_dotenv

# --------------------------------
#            API KEY 
# --------------------------------
# Se carga la clave de API de OpenAI desde un archivo .env
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

x = openai.api_key

embeddings = OpenAIEmbeddings(openai_api_key=x)

# Aqui guardo las bases vectoriales
CHROMA_PATH = "chroma"
DATA_PATH = "Data/MD"



def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

# Aqui cargamos los ducmentos ya convertidos en markdown
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


# Como los docs son muy grandes hay que dividirlos en partes pequesñas q puedan ser referenciadas.

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(

        # Defino cuantos character quiero en cada chunk
        chunk_size=300,

        # Cada chunk va a tener un over lap de 100 characters
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)
    # Esto es si quiero utilizar el servicio de lambda de AWS
    # {'source': 'data/aws_lambda_docs/configuration-tags.md','start_index':690}

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    # De esta forma se borra la base de datos anterior y se deja la ultima
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Para poder buscar entre los chinks se necesita Chroma DB
    # Asi se convierte en una base de datos vectorial que despues se puede modificar
    # Asi es como creo la base de datos con todos los chunks , utiliza los vector embbeding como la KEY
    # Create a new DB from the documents.
    db = Chroma.from_documents(
        # Aqui es donde utilizo el API key para connectar con las funciones de embbeding
        # Es necesario tener una cuenta de OpenAI que se utiliza como la llave

        # Es muy importante tener el PATH definido 
        # SI se quiere subir la base de datos en la nube o utilizarla en otro lado se puede acceder facilmente
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    # La base de datos se deberia guardar sola, pero se puede utilizar el mettdo persist para forzarla
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()