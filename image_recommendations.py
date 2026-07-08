import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import faiss
import pickle
import numpy as np
import streamlit as st

st.set_page_config(
    page_title = "Image based recommendation",
    page_icon = "🖼️",
    layout = "wide"
    )

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("style.css")

st.markdown('<p class="main-header">🖼️ Image Based Recommendation System</p>', unsafe_allow_html=True)

@st.cache_resource
def stored_resources():
    with open("image_paths_lookup.pkl", "rb") as f:
      products_path_lookup = pickle.load(f)    

    index = faiss.read_index("abo_product_vectors.index")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    resnet_model.fc = nn.Identity()
    resnet_model = resnet_model.to(device)
    resnet_model.eval()

    return products_path_lookup, index, device, resnet_model

products_path_lookup, index, device, resnet_model = stored_resources()

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with st.expander("🗂️ Project Highlights", expanded = False):
    st.markdown("<br>", unsafe_allow_html = True)
    
    meta_cols = st.columns(4)
    meta_cols[0].metric(label = "Feature Extractor", value = "ResNet-50", delta = "Pre-trained")
    meta_cols[1].metric(label = "Embedding Dimensions", value = "2048", delta = "Global Avg Pooling")
    meta_cols[2].metric(label = "Vector Search Database", value = "FAISS", delta = "Local FlatIP Index")
    meta_cols[3].metric(label = "Local Custom Dataset", value = "10k Images", delta = "Sampled out of 398k images ")

    st.markdown("<br>", unsafe_allow_html = True)
    
    st.markdown("""
    | Parameter | Details |
    | :--- | :--- |
    | **Dataset Used** | Amazon ABO (Amazon Berkeley Objects) Dataset |
    | **Transformation** | Resize(224, 224),  ToTensor(),  Normalization |
    | **Fully Connected Layer** | Stripped Away (nn.Identity()) |
    | **Evaluation Metric** | Cosine Similarity |
    """)

st.markdown("<br>", unsafe_allow_html = True)

tab_visual, tab_metrics = st.tabs(["💻 VISUAL RECOMMENDATION WINDOW", "📈 EVALUATION METRICS WINDOW"])

with tab_visual:
    st.markdown("<br>", unsafe_allow_html = True)
    
    Product_catalogue, Customer_search, Recommendations = st.columns(3, border = True)

    
    with Product_catalogue:
        st.markdown('<p class="custom-subheader">Product Catalogue</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html = True)
        query_path = []
        for i in range(0, 9000,  180):
            query_path.append(products_path_lookup[i])

        if "product_selected" not in st.session_state:
            st.session_state["product_selected"] = query_path[0]

        grid_columns = st.columns(5)

        for i, path in enumerate(query_path):
            column_index = i % 5

            with grid_columns[column_index]:
                st.image(path)

                if st.button("click", key = f"select_button_{i}"):
                    st.session_state["product_selected"] = path
                    st.rerun()


    with Customer_search:
        user_input = st.session_state["product_selected"]

        st.markdown('<p class="custom-subheader">Searched Product</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html = True)
        st.image(user_input)

    
    with Recommendations:
        query_image = Image.open(user_input).convert("RGB")
        
        query_embedding = resnet_transform(query_image)
        
        query_tensor = query_embedding.unsqueeze(0).to(device)

        with torch.no_grad():
            vector_embedding_tensor = resnet_model(query_tensor)

        vector_embedding = vector_embedding_tensor.cpu().numpy().astype("float32")

        faiss.normalize_L2(vector_embedding)

        distance, indices = index.search(vector_embedding, k = 7)

        all_recommended_path = []

        for idx in indices[0]:
            recommended_path = products_path_lookup[idx]
            if recommended_path == user_input:
                continue
            all_recommended_path.append(recommended_path)

        st.markdown('<p class="custom-subheader">Similar Products</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html = True)

        for score, path in zip(distance[0][:len(all_recommended_path)], all_recommended_path):
            st.write(f"Similarity Score: {score:.2%}")
            st.image(path)            

with tab_metrics:
    st.markdown('<p class="custom-subheader">Evaluation Metrics</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html = True)
            
    k = 5
    sample_queries = 200
    similarity_thrushold = 0.75
    
    all_vectors = []

    for i in range(index.ntotal):
        catalogue_voctors = all_vectors.append(index.reconstruct(i))
    catalogue_matrix = np.vstack(all_vectors).astype("float32")

    visual_precisions = []
    all_cosine_scores = []
    all_recommended_indices = set()

    np.random.seed(42)
    query_indices = np.random.choice(index.ntotal, size = sample_queries, replace = False)

    for query_idx in query_indices:
        query_vector = catalogue_matrix[query_idx].reshape(1, -1)

        distances, indices = index.search(query_vector, k + 1)

        recommended_scores = distances[0][1:]
        recommended_indices = indices[0][1:]

        visual_relevant_found = np.sum(recommended_scores >= similarity_thrushold)
        visual_precision_at_k = visual_relevant_found / k
        visual_precisions.append(visual_precision_at_k)

        all_cosine_scores.extend(recommended_scores)
        all_recommended_indices.update(recommended_indices)

    avg_visual_precision = np.mean(visual_precisions)
    avg_cosine_similarity = np.mean(all_cosine_scores)

    unique_items_recommended = len(all_recommended_indices)
    total_catalogue_size = index.ntotal
    catalogue_coverage = unique_items_recommended / total_catalogue_size
        
        
    kpi_columns = st.columns(2)
    kpi_columns[0].metric(
        label = "Average Visual Precision@K", 
        value = f'{avg_visual_precision:.2%}'
        )
    kpi_columns[1].metric(
        label = "Average Cosine Similarity@K", 
        value = f'{avg_cosine_similarity:.2%}'
        )

    st.markdown("<br>", unsafe_allow_html = True)
    st.markdown("<br>", unsafe_allow_html = True)
    st.markdown(
        '<p class="custom-text">ℹ️ Catalogue Coverage and Unique Recommendations'
        '(For Given K = 5 and Sample Queries = 200 out of 10k)</p>', 
        unsafe_allow_html=True
        )

    kpi_columns = st.columns(2)
    kpi_columns[0].metric(
        label = "Catalogue Coverage by Model", 
        value = f'{catalogue_coverage:.2%}'
        )
    kpi_columns[1].metric(
        label = "Unique Items Utilized by Engine out of '10K' Products", 
        value = f'{unique_items_recommended} Nos.'
        )
