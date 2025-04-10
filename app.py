import streamlit as st
import requests
import json
import numpy as np

# Настройка Mistral AI API
MISTRAL_API_KEY = 'uQwjntCIJ9omN9z8jLTV1VOUvYlbaDIv'
MISTRAL_API_URL = 'https://api.mistral.ai/v1/embeddings'

# Загрузка данных о курсах
try:
    with open('courses.json', 'r', encoding='utf-8') as file:
        courses = json.load(file)
except FileNotFoundError:
    st.error("Файл courses.json не найден!")
    courses = []

# Функция для получения эмбеддингов
def get_embeddings(texts):
    headers = {
        'Authorization': f'Bearer {MISTRAL_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'mistral-embed',
        'input': texts
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        response_json = response.json()
        
        if 'data' not in response_json:
            st.error(f"Неожиданный формат ответа от API: {response_json}")
            return []
        
        embeddings = [item['embedding'] for item in response_json['data']]
        return embeddings
    
    except requests.RequestException as e:
        st.error(f"Ошибка при запросе к API: {str(e)}")
        return []
    except (KeyError, TypeError) as e:
        st.error(f"Ошибка обработки ответа API: {str(e)}")
        return []

def search_courses(query, courses):
    if not courses or not query:
        st.warning("Нет курсов для поиска или запрос пустой")
        return []
    
    descriptions = [course['description'] for course in courses]
    
    query_embeddings = get_embeddings([query])
    if not query_embeddings:
        return []
    query_embedding = query_embeddings[0]
    
    course_embeddings = get_embeddings(descriptions)
    if not course_embeddings:
        return []
    
    similarities = np.dot(course_embeddings, query_embedding) / (
        np.linalg.norm(course_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    similarities = np.nan_to_num(similarities)
  
    sorted_indices = np.argsort(similarities)[::-1]
    recommended_courses = [courses[i] for i in sorted_indices]
    return recommended_courses

st.title("Рекомендация курсов на Karpov Courses")

query = st.text_input("Введите ваш запрос:", placeholder="Например: Хочу изучить Python")

if query:
    with st.spinner("Поиск подходящих курсов..."):
        recommended_courses = search_courses(query, courses)
    
    if recommended_courses:
        st.subheader("Рекомендуемые курсы:")
        for course in recommended_courses:
            st.markdown(f"### [{course['title']}]({course['url']})")
            st.write(course['description'])
            st.markdown("---")
    else:
        st.warning("Не удалось найти подходящие курсы. Попробуйте другой запрос.")
