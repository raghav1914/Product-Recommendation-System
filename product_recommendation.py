import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD

# Synthetic data for demonstration
data = {
    'uniq_id': ['PID001', 'PID002', 'PID003', 'PID004', 'PID005'],
    'crawl_timestamp': ['2024-02-28 09:00:00', '2024-02-28 09:15:00', '2024-02-28 09:30:00', '2024-02-28 09:45:00', '2024-02-28 10:00:00'],
    'product_url': ['https://www.example.com/product1', 'https://www.example.com/product2', 'https://www.example.com/product3', 'https://www.example.com/product4', 'https://www.example.com/product5'],
    'product_name': ['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Camera'],
    'product_category_tree': ['Electronics > Mobiles', 'Electronics > Laptops', 'Electronics > Audio > Headphones', 'Electronics > Tablets', 'Electronics > Cameras'],
    'pid': ['P001', 'P002', 'P003', 'P004', 'P005'],
    'retail_price': [500, 1000, 50, 300, 800],
    'discounted_price': [450, 950, 40, 250, 750],
    'image': ['product1.jpg', 'product2.jpg', 'product3.jpg', 'product4.jpg', 'product5.jpg'],
    'is_FK_Advantage_product': [True, False, True, False, True],
    'description': ['High-end smartphone', 'Powerful laptop', 'Noise-cancelling headphones', 'Portable tablet', 'High-resolution camera'],
    'product_rating': [4.5, 4.8, 4.2, 4.6, 4.7],
    'overall_rating': [4.6, 4.9, 4.3, 4.7, 4.8],
    'brand': ['Brand1', 'Brand2', 'Brand3', 'Brand4', 'Brand5'],
    'product_specifications': ['Specs1', 'Specs2', 'Specs3', 'Specs4', 'Specs5']
}

# Create DataFrame
df = pd.DataFrame(data)

# Content-Based Filtering

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform product descriptions
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

# Compute similarity scores
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Collaborative Filtering

# Prepare the user-item rating matrix for collaborative filtering
ratings_matrix = df.pivot_table(values='product_rating', index='uniq_id', columns='pid', fill_value=0)

# Apply SVD to reduce dimensionality
n_components = min(ratings_matrix.shape) - 1  # Adjust number of components
svd = TruncatedSVD(n_components=n_components, random_state=42)
ratings_svd = svd.fit_transform(ratings_matrix)

# Calculate similarity scores based on SVD-transformed ratings
svd_cosine_similarities = linear_kernel(ratings_svd, ratings_svd)

def get_product_recommendations(product_id, n=5):
    # Check if the product ID exists in the DataFrame
    if product_id not in df['pid'].values:
        messagebox.showerror("Error", "Product ID '{}' not found.".format(product_id))
        return []

    # Get the index of the product in the DataFrame
    product_index = df[df['pid'] == product_id].index[0]
    
    # Get pairwise similarity scores with other products
    sim_scores = list(enumerate(cosine_similarities[product_index]))

    # Sort products by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar products (excluding the product itself)
    top_similar_products = sim_scores[1:n+1]

    # Get indices of top similar products
    similar_product_indices = [i[0] for i in top_similar_products]

    # Return recommended product names
    recommended_product_names = df['product_name'].iloc[similar_product_indices]
    return recommended_product_names.tolist()

def show_recommendations():
    product_id = product_entry.get()
    recommendations = get_product_recommendations(product_id)
    if recommendations:
        recommendations_text.delete(1.0, tk.END)
        recommendations_text.insert(tk.END, "\n".join(recommendations))
        
        # Display product details
        product_details = df[df['pid'] == product_id]
        details_text.delete(1.0, tk.END)
        details_text.insert(tk.END, product_details.to_string(index=False))
    else:
        recommendations_text.delete(1.0, tk.END)
        details_text.delete(1.0, tk.END)

# Create Tkinter window and widgets
window = tk.Tk()
window.title("Flipkart Recommendation System")

# Add Flipkart logo
logo_image = Image.open("download.jpg")  # Replace with the path to your Flipkart logo image
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(window, image=logo_photo)
logo_label.image = logo_photo
logo_label.pack()

product_label = tk.Label(window, text="Enter Product ID:")
product_label.pack()

product_entry = tk.Entry(window)
product_entry.pack()

recommend_button = tk.Button(window, text="Get Recommendations", command=show_recommendations)
recommend_button.pack()

recommendations_label = tk.Label(window, text="Recommended Products:")
recommendations_label.pack()

recommendations_text = scrolledtext.ScrolledText(window, width=50, height=10)
recommendations_text.pack()

details_label = tk.Label(window, text="Product Details:")
details_label.pack()

details_text = scrolledtext.ScrolledText(window, width=50, height=10)
details_text.pack()

window.mainloop()
