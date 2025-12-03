import streamlit as st
from recommender import RecipeRecommender
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Recipe Recommender",
    page_icon="🍳",
    layout="wide"
)

# Initialize the recommender
@st.cache_resource
def load_recommender():
    return RecipeRecommender('train.json')

try:
    recommender = load_recommender()
    total_recipes = len(recommender.df)
except Exception as e:
    st.error(f"Error loading recipes: {e}")
    st.stop()

# App title and description
st.title("🍳 Ingredient-Based Recipe Recommendation System")
st.markdown(f"""
Welcome! Enter the ingredients you have, and I'll recommend the best recipes for you.
The system uses **TF-IDF vectorization** and **cosine similarity** to find the most relevant matches.

**Dataset**: {total_recipes} recipes from multiple cuisines
""")

# Sidebar for filters
st.sidebar.header("Filters")

# Cuisine filter
cuisines = sorted(recommender.df['cuisine'].unique())
selected_cuisines = st.sidebar.multiselect(
    "Filter by cuisine:",
    cuisines,
    default=[]
)

# Allergen filter
st.sidebar.subheader("Dietary Restrictions")
show_all = st.sidebar.checkbox("Show all recipes", value=True)

if not show_all:
    filter_options = st.sidebar.multiselect(
        "Exclude allergens:",
        ["eggs", "dairy", "nuts", "soy"],
        default=[]
    )
else:
    filter_options = []

# Main input area
st.header("Enter Your Ingredients")
user_input = st.text_area(
    "Type ingredients separated by commas (e.g., chicken, garlic, tomatoes, onion):",
    height=100,
    placeholder="tomatoes, garlic, olive oil, basil"
)

# Number of recommendations
col1, col2 = st.columns(2)
with col1:
    num_recipes = st.slider("Number of recipes to show:", min_value=1, max_value=20, value=5)
with col2:
    st.metric("Total Recipes in Dataset", total_recipes)

# Search button
if st.button("🔍 Find Recipes", type="primary"):
    if user_input.strip():
        with st.spinner("Searching for the best recipes..."):
            # Get recommendations
            recommendations = recommender.recommend(
                user_input,
                top_n=num_recipes * 3,  # Get more to filter
                exclude_allergens=filter_options
            )
            
            # Filter by cuisine if selected
            if selected_cuisines:
                recommendations = [r for r in recommendations if r['cuisine'] in selected_cuisines]
            
            # Limit to requested number
            recommendations = recommendations[:num_recipes]
            
            if recommendations:
                st.success(f"Found {len(recommendations)} recipes for you!")
                
                # Display results
                for idx, recipe in enumerate(recommendations, 1):
                    with st.expander(f"#{idx} - {recipe['recipe']} (Match: {recipe['similarity']:.1%})"):
                        # Recipe header
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Cuisine:** {recipe['cuisine'].replace('_', ' ').title()}")
                            st.markdown(f"**Recipe ID:** {recipe['id']}")
                        
                        with col2:
                            allergens = recipe['allergens']
                            if allergens and allergens.lower() != 'none':
                                st.warning(f"⚠️ {allergens}")
                            else:
                                st.success("✅ No allergens")
                        
                        # Ingredients section
                        st.markdown("**Ingredients:**")
                        
                        # Display as columns for better readability
                        ingredients_list = recipe['ingredients_list']
                        
                        # Split into 3 columns
                        cols = st.columns(3)
                        for i, ingredient in enumerate(ingredients_list):
                            with cols[i % 3]:
                                st.write(f"• {ingredient}")
                        
                        # Similarity score visualization
                        st.markdown("**Match Score:**")
                        st.progress(recipe['similarity'])
            else:
                st.warning("No recipes found. Try different ingredients or remove some filters.")
    else:
        st.error("Please enter at least one ingredient!")

# Show some example searches
st.divider()
st.subheader("Need inspiration? Try these:")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("🍝 Italian ingredients"):
        st.session_state.example_input = "tomatoes, basil, garlic, olive oil, parmesan"

with example_col2:
    if st.button("🌮 Mexican ingredients"):
        st.session_state.example_input = "tomatoes, onion, chili, cumin, cilantro"

with example_col3:
    if st.button("🍛 Asian ingredients"):
        st.session_state.example_input = "soy sauce, ginger, garlic, rice, sesame"

# Footer with information
st.divider()
st.markdown("""
### How it works:
1. **TF-IDF Vectorization**: Converts ingredient lists into numerical vectors that capture the importance of each ingredient
2. **Cosine Similarity**: Measures how similar your ingredients are to each recipe (0 = no match, 1 = perfect match)
3. **Ranking**: Shows you the top matches based on similarity scores
4. **Allergen Detection**: Automatically detects common allergens (eggs, dairy, nuts, soy) from ingredients

### Features:
- ✅ Multiple cuisine types (Greek, Italian, Mexican, Chinese, Indian, and more)
- ✅ Automatic allergen detection
- ✅ Dietary filtering options
- ✅ Real-time similarity scoring
""")
