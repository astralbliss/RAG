import streamlit as st
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Настройка Mistral AI API
MISTRAL_API_KEY = 'uQwjntCIJ9omN9z8jLTV1VOUvYlbaDIv'
MISTRAL_API_URL = 'https://api.mistral.ai/v1/embeddings'

# Загрузка данных о курсах
with open('courses.json', 'r', encoding='utf-8') as file:
    courses = json.load(file)

# Функция для получения эмбеддингов
def get_embeddings(texts):
    headers = {
        'Authorization': f'Bearer {MISTRAL_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'input': texts
    }
    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    embeddings = response.json()['data']
    return embeddings

# Функция для поиска курсов
def search_courses(query, courses):
    descriptions = [course['description'] for course in courses]
    query_embedding = get_embeddings([query])[0]
    course_embeddings = get_embeddings(descriptions)

    similarities = cosine_similarity([query_embedding], course_embeddings)[0]
    sorted_indices = similarities.argsort()[::-1]

    recommended_courses = [courses[i] for i in sorted_indices]
    return recommended_courses

# Интерфейс Streamlit
st.title("Рекомендация курсов на Karpov Courses")

query = st.text_input("Введите ваш запрос:")

if query:
    recommended_courses = search_courses(query, courses)
    st.write("Рекомендуемые курсы:")
    for course in recommended_courses:
        st.write(f"[{course['title']}]({course['url']})")
        st.write(course['description'])
        st.write("---")