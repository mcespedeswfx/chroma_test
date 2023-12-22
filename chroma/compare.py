from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
import openai
import os
#----------- ENV VAR -----------------
from dotenv import load_dotenv, find_dotenv



# Lo interesante es ver la distancia que existe entre dos vectores dentro de la base de datos eso dara la similarity
# Los numeros de cada dato somo muy grandes
# La distancia es dificil de calcular 
# Pero OpenAi da una utilizada que calcula la distancia directamente
# Esa es la funcion EVALUATOR

# --------------------------------
#            API KEY 
# --------------------------------
# Se carga la clave de API de OpenAI desde un archivo .env
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
x1 = openai.api_key
embeddings = OpenAIEmbeddings(openai_api_key=x1)

def main():
    # Get embedding for a word.
    embedding_function = embeddings
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance")

    # Esta es una asociacion interesante 
    # No se esta comprando apple como fruta sino com ola compañia
    # Poero eso la similitud con iphone es mucho mas grande
    # Si se compara apple con orange existe una gran distancia
    words = ("apple", "iphone")

    # Lo que se hace es comparar la distancia dde dos palabras de la dupla words
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()