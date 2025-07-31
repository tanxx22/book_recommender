import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
books = pd.read_csv("books.csv")
ratings = pd.read_csv("ratings.csv")

st.title("📚 Simple Book Recommender (No ML, Just Math)")

# Merge books and ratings
book_ratings = pd.merge(ratings, books, on='book_id')

# Create pivot table (user-book matrix)
user_book_matrix = book_ratings.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

# Calculate cosine similarity between books
book_similarity = cosine_similarity(user_book_matrix.T)
book_similarity_df = pd.DataFrame(book_similarity, index=user_book_matrix.columns, columns=user_book_matrix.columns)

# Select a book title for recommendation
book_list = user_book_matrix.columns.tolist()
selected_book = st.selectbox("Select a book you like:", book_list)

if st.button("Recommend Similar Books"):
    st.subheader("📖 You may also like:")
    similar_books = book_similarity_df[selected_book].sort_values(ascending=False)[1:6]
    for i, (title, score) in enumerate(similar_books.items(), start=1):
        st.write(f"{i}. **{title}** (Similarity Score: {score:.2f})")
