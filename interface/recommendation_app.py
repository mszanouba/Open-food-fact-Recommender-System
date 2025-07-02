import streamlit as st
import json
import re
from typing import List, Dict, Any
import os
import sys
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from textblob import TextBlob
import requests
from PIL import Image
import base64
from io import BytesIO
import streamlit.components.v1 as components

os.environ["PYSPARK_PYTHON"] = "C:\\Users\\LENOVO\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"  

# PySpark imports
try:
    from pyspark import SparkConf
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.feature import Tokenizer, HashingTF, MinHashLSH, StopWordsRemover, CountVectorizer, IDF, VectorAssembler
    from pyspark.ml.linalg import Vectors, VectorUDT
    import pyspark.sql.functions as F
    from pyspark.sql.functions import col, concat_ws, explode, lit, udf, array_join
    from pyspark.sql.types import DoubleType, StringType, IntegerType, StructType, StructField
    import numpy as np
    from pyspark.ml.feature import BucketedRandomProjectionLSH
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    st.error("PySpark not available. Please install PySpark to use the recommendation system.")

# Page configuration
st.set_page_config(
    page_title="Food Recommendation System", 
    page_icon="🍽️", 
    initial_sidebar_state="collapsed",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .country-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 0;
        border-radius: 15px;
        border: 3px solid #cbd5e0;
        text-align: center;
        margin: 5px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        height: 200px;
        cursor: pointer;
    }
    .country-card:hover {
        border-color: #4a5568;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        transform: translateY(-3px);
    }
    .country-card.selected {
        border-color: #2d3748;
        background: linear-gradient(135deg, #edf2f7 0%, #cbd5e0 100%);
        box-shadow: 0 12px 30px rgba(45, 55, 72, 0.3);
        transform: translateY(-3px);
    }
    .country-flag {
        width: 100%;
        height: 120px;
        object-fit: cover;
        border-radius: 12px 12px 0 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .country-info {
        padding: 15px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
    }
    .ingredient-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%) !important;
        color: white !important;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4a5568;
        margin: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .ingredient-card h4 {
        color: white !important;
        margin-bottom: 8px;
    }
    .ingredient-card p {
        color: #e2e8f0 !important;
        margin-bottom: 5px;
    }
    .ingredient-card strong {
        color: #ffd700 !important;
    }
    .product-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .product-card:hover {
        border-color: #4a5568;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .product-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .product-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2d3748;
        margin-bottom: 10px;
    }
    .product-info {
        background: #f7fafc;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #4a5568;
    }
    .nutri-score {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        margin: 2px;
    }
    .nutri-a { background-color: #4CAF50; }
    .nutri-b { background-color: #8BC34A; }
    .nutri-c { background-color: #FFEB3B; color: black; }
    .nutri-d { background-color: #FF9800; }
    .nutri-e { background-color: #F44336; }
    .stButton > button {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .main-header {
        background: linear-gradient(135deg, #DFD0B8 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .recommend-button {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 15px 30px !important;
        border-radius: 25px !important;
        font-size: 18px !important;
        transition: 0.3s ease-in-out !important;
        margin: 20px 0 !important;
    }
    .recommend-button:hover {
        opacity: 0.9 !important;
        transform: scale(1.05) !important;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = 0
if 'ingredients' not in st.session_state:
    st.session_state.ingredients = []
if 'recipe_text' not in st.session_state:
    st.session_state.recipe_text = ""
if 'show_filters' not in st.session_state:
    st.session_state.show_filters = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'spark_session' not in st.session_state:
    st.session_state.spark_session = None
if 'recommender' not in st.session_state:
    st.session_state.recommender = None

# Countries data with real flag URLs
COUNTRIES = {
    "France": {
        "flag": "🇫🇷", 
        "flag_url": "https://flagcdn.com/w160/fr.png",
        "cuisine": "French", 
        "specialties": ["croissant", "baguette", "cheese", "wine"]
    },
    "Germany": {
        "flag": "🇩🇪", 
        "flag_url": "https://flagcdn.com/w160/de.png",
        "cuisine": "German", 
        "specialties": ["sausage", "beer", "pretzel", "sauerkraut"]
    },
    "United Kingdom": {
        "flag": "🇬🇧", 
        "flag_url": "https://flagcdn.com/w160/gb.png",
        "cuisine": "British", 
        "specialties": ["fish", "chips", "tea", "pudding"]
    },
    "United State of America": {
        "flag": "🇺🇸", 
        "flag_url": "https://flagcdn.com/w160/us.png",
        "cuisine": "American", 
        "specialties": ["burger", "fries", "barbecue", "apple pie"]
    }
}

# Ingredient categories and allergens
VALID_INGREDIENTS_FOR_RECOMMENDATIONS = [
    # Baking ingredients
    'butter', 'sugar', 'eggs', 'vanilla', 'cocoa powder', 'flour', 'salt', 
    'baking powder', 'nuts', 'chocolate chips', 'brown sugar', 'white sugar', 
    'olive oil', 'onion', 'garlic', 'ground beef', 'beef', 'tomatoes', 
    'tomato paste', 'oregano', 'basil', 'parsley', 'parmesan', 'cheese',
    'spaghetti', 'pasta', 'brownies','yaourt',
    'pepper', 'black pepper', 'paprika', 'cumin', 'thyme', 'rosemary',
    'cinnamon', 'nutmeg', 'ginger', 'turmeric', 'bay leaves',
    'milk', 'cream', 'yogurt', 'chicken', 'fish', 'salmon', 'tuna',
    'pork', 'lamb', 'turkey',
    'carrot', 'potato', 'onion', 'bell pepper', 'mushroom', 'spinach',
    'broccoli', 'cauliflower', 'zucchini', 'eggplant', 'apple', 'banana',
    'lemon', 'lime', 'orange',
    'rice', 'quinoa', 'oats', 'barley', 'beans', 'lentils', 'chickpeas',
    'vegetable oil', 'canola oil', 'coconut oil', 'vinegar', 'balsamic vinegar',
    'yeast', 'baking soda', 'cornstarch', 'honey', 'maple syrup',
    'soy sauce', 'worcestershire sauce', 'hot sauce', 'mustard', 'ketchup',
    'chocolat', 'chocolate', 
    'beurre',
    'farine',
    'levure', 
    'jaunes', 'jaunes d\'oeufs', 'jaune d\'oeuf',
    'blancs', 'blancs d\'oeufs', 'blanc d\'oeuf',
    'sucre',
    'oeufs', 'oeuf', 
    'sel',
    'baking powder', 'levure chimique', 
    'egg whites', 'egg yolks',
    'dark chocolate', 'chocolat noir',
    'milk chocolate', 'chocolat au lait',
    'unsalted butter', 'beurre doux',
    'salted butter', 'beurre salé',
    'all purpose flour', 'farine tout usage',
    'plain flour', 'farine ordinaire',
    'huile', 'huile d\'olive',
    'tomates', 'tomate',
    'oignon', 'ail', 
    'viande', 'boeuf', 'porc', 'agneau',
    'fromage', 'parmesan', 'gruyère',
    'lait', 'crème', 'yaourt',
    'pâtes', 'spaghetti', 'macaroni',
    'riz', 'quinoa', 'blé',
    'épices', 'herbes', 'basilic', 'persil',
    'poivre', 'poivre noir', 
    'miel', 'sirop d\'érable'
]

INGREDIENT_CATEGORIES = {
    'flour': 'grains', 'bread': 'grains', 'rice': 'grains', 'pasta': 'grains',
    'eggs': 'protein', 'chicken': 'protein', 'beef': 'protein', 'fish': 'protein', 'turkey': 'protein',
    'milk': 'dairy', 'butter': 'dairy', 'cheese': 'dairy', 'cream': 'dairy', 'yogurt': 'dairy',
    'oil': 'fats', 'olive oil': 'fats', 'coconut oil': 'fats',
    'tomatoes': 'vegetables', 'onion': 'vegetables', 'garlic': 'vegetables', 'carrots': 'vegetables',
    'potatoes': 'vegetables', 'bell pepper': 'vegetables', 'spinach': 'vegetables',
    'salt': 'seasonings', 'pepper': 'seasonings', 'herbs': 'seasonings', 'spices': 'seasonings',
    'sugar': 'sweeteners', 'honey': 'sweeteners', 'maple syrup': 'sweeteners'
}

ALLERGENS = {
    'flour': ['gluten'], 'bread': ['gluten'], 'pasta': ['gluten'],
    'eggs': ['eggs'], 'milk': ['dairy'], 'butter': ['dairy'], 
    'cheese': ['dairy'], 'cream': ['dairy'], 'yogurt': ['dairy'],
    'fish': ['fish'], 'nuts': ['nuts'], 'peanuts': ['peanuts']
}

# ============= PYSPARK RECOMMENDATION SYSTEM =============

class ProductRecommendationSystem:
    """
    Complete product recommendation system using PySpark
    Recommendations based on product_name and categories
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
        self.df = None
        self.vectorized_df = None
        self.pipeline_model = None
        self.lsh_model = None
        
    def load_data(self, json_path):
        """Load data from JSON file"""
        print("Loading data...")
        self.df = self.spark.read.format("json").load(json_path)
        
        # Filter out records with null product names
        self.df = self.df.filter(col("product_name").isNotNull())
        
        # Add row number as ID since no ID column exists
        from pyspark.sql.window import Window
        from pyspark.sql.functions import row_number
        
        window = Window.orderBy(col("code"))
        self.df = self.df.withColumn("id", row_number().over(window))
        
        print(f"Loaded {self.df.count()} products")
        return self
    
    def prepare_features(self):
        """Prepare text features for vectorization"""
        print("Preparing features...")
        
        # Handle categories array - convert to string
        df_with_categories = self.df.withColumn(
            "categories_text", 
            array_join(col("categories"), " ")
        )
        
        # Combine product name and categories for better recommendations
        df_with_features = df_with_categories.withColumn(
            "combined_text",
            concat_ws(" ", col("product_name"), col("categories_text"))
        )
        
        # Fill null values with empty strings
        df_with_features = df_with_features.fillna({
            "combined_text": "",
            "product_name": "",
            "categories_text": ""
        })
        
        self.df = df_with_features
        return self
    
    def build_pipeline(self, vocab_size=10000, min_df=2):
        """Build ML pipeline for text vectorization"""
        print("Building ML pipeline...")
        
        # Tokenization
        tokenizer = Tokenizer(inputCol="combined_text", outputCol="words")
        
        # Remove stop words
        stop_words_remover = StopWordsRemover(
            inputCol="words", 
            outputCol="filtered_words"
        )
        
        # Count Vectorizer
        count_vectorizer = CountVectorizer(
            inputCol="filtered_words",
            outputCol="raw_features",
            vocabSize=vocab_size,
            minDF=min_df
        )
        
        # TF-IDF
        idf = IDF(
            inputCol="raw_features",
            outputCol="features"
        )
        
        # Create pipeline
        self.pipeline = Pipeline(stages=[
            tokenizer,
            stop_words_remover,
            count_vectorizer,
            idf
        ])
        
        return self
    
    def train_model(self):
        """Train the pipeline model"""
        print("Training model...")
        
        # Fit the pipeline
        self.pipeline_model = self.pipeline.fit(self.df)
        
        # Transform the data
        self.vectorized_df = self.pipeline_model.transform(self.df)
        
        print("Model training completed!")
        return self
    
    def build_lsh_index(self, num_hash_tables=5):
        """Build LSH index for fast similarity search"""
        print("Building LSH index...")
        
        # Create LSH model
        lsh = BucketedRandomProjectionLSH(
            inputCol="features",
            outputCol="hashes",
            bucketLength=2.0,
            numHashTables=num_hash_tables
        )
        
        # Fit LSH model
        self.lsh_model = lsh.fit(self.vectorized_df)
        
        # Transform data with LSH
        self.vectorized_df = self.lsh_model.transform(self.vectorized_df)
        
        print("LSH index built successfully!")
        return self
    
    def vectorize_query(self, product_name, categories=None):
        """Vectorize a query product"""
        if categories is None:
            categories = []
        
        # Convert categories to string
        categories_text = " ".join(categories) if categories else ""
        
        # Create combined text
        combined_text = f"{product_name} {categories_text}".strip()
        
        # Create DataFrame
        query_df = self.spark.createDataFrame([{
            "combined_text": combined_text,
            "product_name": product_name,
            "categories_text": categories_text
        }])
        
        # Transform using the pipeline
        query_vectorized = self.pipeline_model.transform(query_df)
        
        return query_vectorized.select("features").collect()[0]["features"]
    
    def cosine_similarity_udf(self, query_vector):
        """Create UDF for cosine similarity calculation"""
        
        def cosine_sim(vector):
            if vector is None:
                return 0.0
            
            try:
                v1 = vector.toArray()
                v2 = query_vector.toArray()
                
                dot_product = float(np.dot(v1, v2))
                norm1 = float(np.linalg.norm(v1))
                norm2 = float(np.linalg.norm(v2))
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return dot_product / (norm1 * norm2)
            except:
                return 0.0
        
        return udf(cosine_sim, DoubleType())
    
    def get_filtered_recommendations(self, product_name, categories=None, filters=None, num_recommendations=10):
        """Get recommendations with filtering"""
        try:
            # Vectorize query
            query_vector = self.vectorize_query(product_name, categories)
            
            # Create cosine similarity UDF
            cosine_udf = self.cosine_similarity_udf(query_vector)
            
            # Calculate similarities
            df_with_similarity = self.vectorized_df.withColumn(
                "similarity", 
                cosine_udf(col("features"))
            )
            
            # Apply filters if provided
            filtered_df = df_with_similarity
            
            if filters:
                # Apply dietary filters
                if 'gluten_free' in filters and filters['gluten_free']:
                    filtered_df = filtered_df.filter(col("contains_gluten") == 0)
                
                if 'dairy_free' in filters and filters['dairy_free']:
                    filtered_df = filtered_df.filter(col("contains_milk") == 0)
                
            
            # Filter out exact matches and get top recommendations
            recommendations = filtered_df.filter(
                col("product_name") != product_name
            ).orderBy(
                col("similarity").desc()
            ).limit(num_recommendations)
            
            return recommendations.collect()
            
        except Exception as e:
            print(f"Error in get_filtered_recommendations: {e}")
            return []
    
    def load_model(self, model_path, vectorized_data_path):
        """Load pre-trained model and vectorized data"""
        print("Loading pre-trained model...")
        
        try:
            # Load pipeline model
            self.pipeline_model = PipelineModel.load(model_path)
            
            # Load vectorized data
            self.vectorized_df = self.spark.read.parquet(vectorized_data_path)
            
            # Rebuild LSH if needed
            if 'hashes' not in self.vectorized_df.columns:
                self.build_lsh_index()
            
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# ============= INGREDIENT EXTRACTION CLASSES AND FUNCTIONS =============

class AutoIngredientExtractor:
    def __init__(self):
        print("Loading models...")
        
        try:
            # Grammar correction model
            self.grammar_corrector = pipeline(
                "text2text-generation",
                model="vennify/t5-base-grammar-correction"
            )
            
            # Recipe-specific NER model (trained on recipe datasets)
            self.recipe_ner = pipeline(
                "ner",
                model="edwardjross/xlm-roberta-base-finetuned-recipe-all",
                aggregation_strategy="simple"
            )
            
            # Food-specific NER model as backup
            self.food_ner = pipeline(
                "ner", 
                model="Dizex/InstaFoodRoBERTa-NER",
                aggregation_strategy="simple"
            )
            
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to basic extraction
            self.grammar_corrector = None
            self.recipe_ner = None
            self.food_ner = None
    
    def correct_grammar(self, text):
        """Correct grammar using pre-trained model"""
        if not self.grammar_corrector:
            return text
        try:
            result = self.grammar_corrector(f"grammar: {text}", max_length=512)
            return result[0]['generated_text']
        except Exception as e:
            print(f"Grammar correction failed: {e}")
            return text
    
    def extract_with_recipe_ner(self, text):
        """Extract using recipe-specific NER model"""
        if not self.recipe_ner:
            return []
        try:
            entities = self.recipe_ner(text)
            ingredients = []
            
            for entity in entities:
                if entity['score'] > 0.5:  # Confidence threshold
                    ingredient = entity['word'].strip()
                    # Clean up sub-tokens (remove ## prefixes)
                    ingredient = re.sub(r'##', '', ingredient)
                    if len(ingredient) > 2:
                        ingredients.append(ingredient.lower())
            
            return list(set(ingredients))
        except Exception as e:
            print(f"Recipe NER failed: {e}")
            return []
    
    def extract_with_food_ner(self, text):
        """Extract using food-specific NER model"""
        if not self.food_ner:
            return []
        try:
            entities = self.food_ner(text)
            ingredients = []
            
            for entity in entities:
                if entity['entity_group'] == 'FOOD' and entity['score'] > 0.5:
                    ingredient = entity['word'].strip()
                    # Clean up sub-tokens
                    ingredient = re.sub(r'##', '', ingredient)
                    if len(ingredient) > 2:
                        ingredients.append(ingredient.lower())
            
            return list(set(ingredients))
        except Exception as e:
            print(f"Food NER failed: {e}")
            return []
    
    def extract_quantities_regex(self, text, ingredients):
        """Extract quantities for found ingredients using regex"""
        ingredients_with_units = []
        
        for ingredient in ingredients:
            # Look for this ingredient in text with quantities
            pattern = rf'([½¼¾\d\/\.\s]*(?:cup|cups|tbsp|tsp|tablespoon|teaspoon|oz|lb|gram|kg|liter|ml|piece|slice|clove|stick)s?)\s+(?:of\s+)?.*?{re.escape(ingredient)}'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            found_with_quantity = False
            for match in matches:
                quantity = match.group(1).strip()
                if quantity:
                    ingredients_with_units.append(f"{quantity} {ingredient}")
                    found_with_quantity = True
                    break
            
            if not found_with_quantity:
                ingredients_with_units.append(ingredient)
        
        return ingredients_with_units
    
    def process_input(self, user_input):
        """Main processing function using pre-trained models only"""
        print("Correcting grammar...")
        corrected_text = self.correct_grammar(user_input)
        
        print("Extracting ingredients with recipe NER...")
        recipe_ingredients = self.extract_with_recipe_ner(corrected_text)
        
        print("Extracting ingredients with food NER...")  
        food_ingredients = self.extract_with_food_ner(corrected_text)
        
        # Combine results from both models
        all_ingredients = list(set(recipe_ingredients + food_ingredients))
        
        # Remove very short or common words that aren't ingredients
        filtered_ingredients = [
            ing for ing in all_ingredients 
            if len(ing) > 2 and ing not in ['the', 'and', 'or', 'in', 'to', 'for', 'with']
        ]
        
        print("Extracting quantities...")
        ingredients_with_units = self.extract_quantities_regex(corrected_text, filtered_ingredients)
        
        return {
            'original_text': user_input,
            'corrected_text': corrected_text,
            'ingredients_only': filtered_ingredients,
            'ingredients_with_units': ingredients_with_units
        }

# Global extractor instance
_extractor = None

def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = AutoIngredientExtractor()
    return _extractor

def extract_ingredients_fixed(user_input):
    """Fixed automated ingredient extraction"""
    extractor = get_extractor()
    result = extractor.process_input(user_input)
    return result['ingredients_only']

def filter_good_ingredients(ingredients_list):
    # Define common valid ingredients (can be expanded)
    valid_ingredients = {
        'butter', 'sugar', 'eggs', 'vanilla', 'cocoa powder', 'flour', 'salt', 
        'baking powder', 'nuts', 'chocolate chips', 'brown sugar', 'white sugar', 
        'olive oil', 'onion', 'garlic', 'ground beef', 'beef', 'tomatoes', 
        'tomato paste', 'oregano', 'basil', 'parsley', 'parmesan', 'cheese',
        'spaghetti', 'pasta', 'brownies','yaourt',
        'pepper', 'black pepper', 'paprika', 'cumin', 'thyme', 'rosemary',
        'cinnamon', 'nutmeg', 'ginger', 'turmeric', 'bay leaves',
        'milk', 'cream', 'yogurt', 'chicken', 'fish', 'salmon', 'tuna',
        'pork', 'lamb', 'turkey',
        'carrot', 'potato', 'onion', 'bell pepper', 'mushroom', 'spinach',
        'broccoli', 'cauliflower', 'zucchini', 'eggplant', 'apple', 'banana',
        'lemon', 'lime', 'orange',
        'rice', 'quinoa', 'oats', 'barley', 'beans', 'lentils', 'chickpeas',
        'vegetable oil', 'canola oil', 'coconut oil', 'vinegar', 'balsamic vinegar',
        'yeast', 'baking soda', 'cornstarch', 'honey', 'maple syrup',
        'soy sauce', 'worcestershire sauce', 'hot sauce', 'mustard', 'ketchup'
    }
    
    # Filter criteria
    def is_valid_ingredient(ingredient):
        ingredient = ingredient.lower().strip()
        
        if len(ingredient) < 3:
            return False
            
        invalid_fragments = {
            'uns', 'all', 'unsweeten', 'cup', 'ies', 'unsal', 'alted', 
            'bak', 'brown', 'large', 'fud'
        }
        if ingredient in invalid_fragments:
            return False
            
        if len(ingredient) == 1 or ingredient.isdigit():
            return False
            
        units = {'cup', 'cups', 'tsp', 'tbsp', 'oz', 'lb', 'g', 'kg', 'ml', 'l'}
        if ingredient in units:
            return False
            
        if ingredient in valid_ingredients:
            return True
            
        ingredient_keywords = ['powder', 'extract', 'flour', 'sugar', 'oil', 'butter', 'milk']
        if any(keyword in ingredient for keyword in ingredient_keywords):
            return True
            
        if len(ingredient) >= 4 and ingredient.isalpha():
            return True
            
        return False
    
    filtered = [ing for ing in ingredients_list if is_valid_ingredient(ing)]
    
    seen = set()
    unique_filtered = []
    for ing in filtered:
        if ing.lower() not in seen:
            seen.add(ing.lower())
            unique_filtered.append(ing)
    
    return unique_filtered

# ============= SPARK INITIALIZATION =============

@st.cache_resource
def initialize_spark():
    """Initialize Spark session"""
    if not PYSPARK_AVAILABLE:
        return None
    
    try:
        conf = SparkConf() \
            .setAppName("Product Recommendation System") \
            .setMaster("local[*]") \
            .set("spark.executor.memory", "2g") \
            .set("spark.sql.shuffle.partitions", "50") \
            .set("spark.executor.cores", "2") \
            .set("spark.driver.memory", "2g") \
            .set("spark.hadoop.hadoop.native.io.enable", "false")

        spark = SparkSession.builder \
            .config(conf=conf) \
            .getOrCreate()
        
        return spark
    except Exception as e:
        st.error(f"Failed to initialize Spark: {e}")
        return None

@st.cache_resource
def initialize_recommender():
    """Initialize recommendation system"""
    spark = initialize_spark()
    if spark is None:
        return None
    
    try:
        recommender = ProductRecommendationSystem(spark)
        # Try to load existing model
        model_loaded = recommender.load_model(
            "models/recommendation_pipeline_hope", 
            "data/vectorized_products_hope.parquet"
        )
        
        if not model_loaded:
            st.warning("Pre-trained model not found. Please train the model first.")
            return None
        
        return recommender
    except Exception as e:
        st.error(f"Failed to initialize recommender: {e}")
        return None

# ============= STREAMLIT UI FUNCTIONS =============

def extract_ingredients_advanced(recipe_text: str) -> List[Dict]:
    """Advanced ingredient extraction using ML models"""
    try:
        # Use the extraction functions
        raw_ingredients = extract_ingredients_fixed(recipe_text)
        good_ingredients = filter_good_ingredients(raw_ingredients)
        
        # Convert to the expected format for Streamlit app
        formatted_ingredients = []
        for ingredient in good_ingredients:
            formatted_ingredients.append({
                'name': ingredient,
                'category': INGREDIENT_CATEGORIES.get(ingredient.lower(), 'other'),
                'allergens': ALLERGENS.get(ingredient.lower(), [])
            })
        
        return formatted_ingredients
        
    except Exception as e:
        st.error(f"Advanced extraction failed: {e}")
        return []

def filter_ingredients(ingredients: List[Dict], filters: Dict) -> List[Dict]:
    """Filter ingredients based on criteria"""
    filtered = []
    
    for ingredient in ingredients:
        # Check dietary restrictions
        if 'vegetarian' in filters.get('dietary', []) and ingredient['category'] == 'protein' and ingredient['name'] in ['chicken', 'beef', 'fish']:
            continue
        if 'vegan' in filters.get('dietary', []) and ingredient['category'] in ['dairy', 'protein'] and ingredient['name'] != 'fish':
            if ingredient['name'] in ['milk', 'butter', 'cheese', 'cream', 'yogurt', 'eggs', 'chicken', 'beef']:
                continue
        if 'gluten-free' in filters.get('dietary', []) and 'gluten' in ingredient['allergens']:
            continue
        if 'dairy-free' in filters.get('dietary', []) and 'dairy' in ingredient['allergens']:
            continue
        if 'nut-free' in filters.get('dietary', []) and 'nuts' in ingredient['allergens']:
            continue
        if 'egg-free' in filters.get('dietary', []) and 'eggs' in ingredient['allergens']:
            continue
        if 'peanuts-free' in filters.get('dietary', []) and 'peanuts' in ingredient['allergens']:
            continue
        if 'fish-free' in filters.get('dietary', []) and 'fish' in ingredient['allergens']:
            continue
            
        filtered.append(ingredient)
    
    return filtered

def get_product_recommendations(ingredient_name: str, recommender_system, method="cosine", num_recommendations=5):
    """Get product recommendations for a specific ingredient using PySpark"""
    try:
        if recommender_system is None:
            st.warning("Recommendation system not initialized")
            return []
        
        # Get recommendations from PySpark system using the correct method name
        results = recommender_system.get_filtered_recommendations(
            product_name=ingredient_name,
            categories=None,
            filters=None,  # Apply filters separately if needed
            num_recommendations=num_recommendations
        )
        
        # The results are already a list of Row objects, convert to dicts
        recommendations = []
        for row in results:
            rec_dict = row.asDict() if hasattr(row, 'asDict') else dict(row)
            recommendations.append(rec_dict)
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []

def apply_product_filters(recommendations: List[Dict], filters: Dict) -> List[Dict]:
    """Apply filters to product recommendations"""
    if not recommendations:
        return []
    
    filtered_recs = []
    
    for rec in recommendations:
        # Apply dietary filters
        dietary_filters = filters.get('dietary', [])
        
        # Gluten-free filter
        if 'gluten-free' in dietary_filters:
            if rec.get('contains_gluten', 1) != 0:
                continue
        
        # Dairy-free filter
        if 'dairy-free' in dietary_filters:
            if rec.get('contains_milk', 1) != 0:
                continue
        
        # Vegan filter (no animal products)
        if 'vegan' in dietary_filters:
            if (rec.get('contains_milk', 1) != 0 or 
                rec.get('contains_eggs', 1) != 0 or
                'meat' in str(rec.get('categories', '')).lower() or
                'fish' in str(rec.get('categories', '')).lower()):
                continue
        
        # Vegetarian filter (no meat/fish)
        if 'vegetarian' in dietary_filters:
            if ('meat' in str(rec.get('categories', '')).lower() or
                'fish' in str(rec.get('categories', '')).lower()):
                continue
        
        filtered_recs.append(rec)
    
    return filtered_recs

def display_product_recommendations(ingredient_name: str, recommendations: List[Dict]):
    """Display product recommendations in a nice format"""
    if not recommendations:
        st.info(f"No product recommendations found for '{ingredient_name}'")
        return
    
    st.markdown(f"### 🛒 Product Recommendations for **{ingredient_name.title()}**")
    
    # Create columns for product display
    num_cols = min(3, len(recommendations))
    if num_cols > 0:
        cols = st.columns(num_cols)
        
        for i, rec in enumerate(recommendations[:6]):  # Show max 6 products
            col_idx = i % num_cols
            try:
                # Check if it's a PySpark Row
                if hasattr(rec, 'asDict'):
                    rec_dict = rec.asDict()
                # Check if it's already a dictionary
                elif isinstance(rec, dict):
                    rec_dict = rec
                # Check if it's a pandas Series
                elif hasattr(rec, 'to_dict'):
                    rec_dict = rec.to_dict()
                else:
                    st.error(f"Unknown recommendation type: {type(rec)}")
                    continue
                    
            except Exception as e:
                st.error(f"Error processing recommendation: {e}")
                continue
            
            with cols[col_idx]:
                # Product card
                product_name = rec_dict.get('product_name', 'Unknown Product')
                categories = rec_dict.get('categories', [])
                nutriscore = rec_dict.get('nutriscore_grade', 'N/A')
                ecoscore = rec_dict.get('ecoscore_score', 'N/A')
                similarity = rec_dict.get('similarity', 0)
                image_url = rec_dict.get('image_url', '')
                
                # Nutritional information
                carbs = rec_dict.get('carbohydrates_100g')
                is_vegan = rec_dict.get('is_vegan')
                is_vegetarian = rec_dict.get('is_vegetarian')
                nova_group = rec_dict.get('nova_group')
                proteins = rec_dict.get('proteins_100g')
                salt = rec_dict.get('salt_100g')
                saturated_fat = rec_dict.get('saturated-fat_100g')
                serving_info = rec_dict.get('serving_info')
                sugars = rec_dict.get('sugars_100g')
                
                # Format categories
                if isinstance(categories, list):
                    categories_str = ', '.join(categories[:3])  # Show first 3 categories
                else:
                    categories_str = str(categories)
                
                display_name = product_name[:50] + ('...' if len(product_name) > 50 else '')
                
                # Determine NutriScore background color
                if nutriscore in ['A', 'B']:
                    nutri_bg_color = '#00b050'
                elif nutriscore in ['D', 'E']:
                    nutri_bg_color = '#ff6b6b'
                else:
                    nutri_bg_color = '#ffd93d'
                
                # Format similarity score
                if isinstance(similarity, (int, float)):
                    similarity_str = f"{similarity:.3f}"
                else:
                    similarity_str = 'N/A'
                
                # Build nutritional info tooltip content
                nutrition_items = []
                
                if carbs is not None:
                    nutrition_items.append(f"<div class='nutrition-item'><span class='nutrition-label'>Carbohydrates:</span> <span class='nutrition-value'>{carbs:.2f}g/100g</span></div>")
                if proteins is not None:
                    nutrition_items.append(f"<div class='nutrition-item'><span class='nutrition-label'>Proteins:</span> <span class='nutrition-value'>{proteins:.2f}g/100g</span></div>")
                if sugars is not None:
                    nutrition_items.append(f"<div class='nutrition-item'><span class='nutrition-label'>Sugars:</span> <span class='nutrition-value'>{sugars:.2f}g/100g</span></div>")
                if salt is not None:
                    nutrition_items.append(f"<div class='nutrition-item'><span class='nutrition-label'>Salt:</span> <span class='nutrition-value'>{salt:.2f}g/100g</span></div>")
                if saturated_fat is not None:
                    nutrition_items.append(f"<div class='nutrition-item'><span class='nutrition-label'>Saturated Fat:</span> <span class='nutrition-value'>{saturated_fat:.2f}g/100g</span></div>")
                if nova_group is not None:
                    nutrition_items.append(f"<div class='nutrition-item'><span class='nutrition-label'>NOVA Group:</span> <span class='nutrition-value'>{int(nova_group)}</span></div>")
                if serving_info is not None:
                    nutrition_items.append(f"<div class='nutrition-item'><span class='nutrition-label'>Serving:</span> <span class='nutrition-value'>{serving_info}</span></div>")
                if is_vegan is not None:
                    nutrition_items.append(f"<div class='nutrition-item'><span class='nutrition-label'>Vegan:</span> <span class='nutrition-value'>{'✓' if is_vegan else '✗'}</span></div>")
                if is_vegetarian is not None:
                    nutrition_items.append(f"<div class='nutrition-item'><span class='nutrition-label'>Vegetarian:</span> <span class='nutrition-value'>{'✓' if is_vegetarian else '✗'}</span></div>")
                
                nutrition_content = "".join(nutrition_items) if nutrition_items else "<div class='no-nutrition'>No nutritional information available</div>"
                
                card_id = f"card_{hash(product_name)}_{col_idx}"
                
                # Create image HTML - always reserve space for image
                if image_url and image_url.strip():
                    image_html = f'<img src="{image_url.strip()}" alt="{product_name}" class="product-image-{card_id}" onerror="this.parentElement.querySelector(\'.image-placeholder-{card_id}\').style.display=\'flex\'; this.style.display=\'none\';">'
                else:
                    image_html = ""
                
                # Create product card HTML with flip effect
                card_html = f"""
                <style>
                .product-card-container-{card_id} {{
                    perspective: 1000px;
                    height: 400px;
                    margin: 8px 0;
                }}
                
                .product-card-{card_id} {{
                    position: relative;
                    width: 100%;
                    height: 100%;
                    text-align: center;
                    transition: transform 0.6s;
                    transform-style: preserve-3d;
                    cursor: pointer;
                }}
                
                .product-card-{card_id}:hover {{
                    transform: rotateY(180deg);
                }}
                
                .card-face-{card_id} {{
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    backface-visibility: hidden;
                    border: 1px solid #ddd;
                    border-radius: 12px;
                    background: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    display: flex;
                    flex-direction: column;
                }}
                
                .card-back-{card_id} {{
                    transform: rotateY(180deg);
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    justify-content: center;
                    align-items: center;
                }}
                
                .image-section-{card_id} {{
                    height: 140px;
                    position: relative;
                    border-radius: 12px 12px 0 0;
                    overflow: hidden;
                    background-color: #f8f9fa;
                }}
                
                .product-image-{card_id} {{
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }}
                
                .image-placeholder-{card_id} {{
                    width: 100%;
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    color: #666;
                    font-size: 14px;
                    flex-direction: column;
                }}
                
                .content-section-{card_id} {{
                    flex: 1;
                    padding: 16px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                }}
                
                .product-title-{card_id} {{
                    font-size: 16px;
                    font-weight: bold;
                    margin-bottom: 12px;
                    color: #333;
                    line-height: 1.3;
                    min-height: 42px;
                    display: flex;
                    align-items: center;
                }}
                
                .nutriscore-badge-{card_id} {{
                    display: inline-block;
                    background-color: {nutri_bg_color};
                    color: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                    margin: 2px 0;
                }}
                
                .product-info-{card_id} {{
                    font-size: 13px;
                    color: #666;
                    margin: 4px 0;
                    text-align: left;
                }}
                
                .nutrition-content-{card_id} {{
                    height: 100%;
                    overflow-y: auto;
                }}
                
                .nutrition-item {{
                    display: flex;
                    justify-content: space-between;
                    margin: 8px 0;
                    padding: 4px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.2);
                }}
                
                .nutrition-label {{
                    font-weight: 500;
                    opacity: 0.9;
                }}
                
                .nutrition-value {{
                    font-weight: bold;
                }}
                
                .no-nutrition {{
                    text-align: center;
                    opacity: 0.8;
                    font-style: italic;
                }}
                
                .flip-hint-{card_id} {{
                    position: absolute;
                    bottom: 8px;
                    right: 12px;
                    font-size: 11px;
                    color: #999;
                    opacity: 0.7;
                }}
                
                .back-title-{card_id} {{
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 16px;
                    text-align: center;
                    border-bottom: 2px solid rgba(255,255,255,0.3);
                    padding-bottom: 8px;
                }}
                </style>
                
                <div class="product-card-container-{card_id}">
                    <div class="product-card-{card_id}">
                        <!-- Front Face -->
                        <div class="card-face-{card_id}">
                            <div class="image-section-{card_id}">
                                {image_html}
                                <div class="image-placeholder-{card_id}" style="display: {'flex' if not image_html else 'none'};">
                                    <div>📦</div>
                                    <div>No Image</div>
                                </div>
                            </div>
                            
                            <div class="content-section-{card_id}">
                                <div class="product-title-{card_id}">{display_name}</div>
                                
                                <div class="product-info-{card_id}">
                                    <strong>Categories:</strong> {categories_str}
                                </div>
                                
                                <div class="product-info-{card_id}">
                                    <strong>NutriScore:</strong> 
                                    <span class="nutriscore-badge-{card_id}">{nutriscore}</span>
                                </div>
                                
                                <div class="product-info-{card_id}">
                                    <strong>EcoScore:</strong> {ecoscore}
                                </div>
                                
                                <div class="product-info-{card_id}">
                                    <strong>Similarity:</strong> {similarity_str}
                                </div>
                            </div>
                            
                            <div class="flip-hint-{card_id}">Hover for nutrition info</div>
                        </div>
                        
                        <!-- Back Face -->
                        <div class="card-face-{card_id} card-back-{card_id}">
                            <div class="back-title-{card_id}">Nutritional Information</div>
                            <div class="nutrition-content-{card_id}">
                                {nutrition_content}
                            </div>
                        </div>
                    </div>
                </div>
                """
                
                components.html(card_html, height=420)
                            
                            
def initialize_recommendation_system():
    """Initialize the PySpark recommendation system"""
    try:
        if 'recommender_system' not in st.session_state:
            with st.spinner("Initializing recommendation system..."):
                # Get Spark session first
                spark = initialize_spark()
                if spark is None:
                    st.error("Failed to initialize Spark session")
                    st.session_state.recommender_system = None
                    return None
                
                # Initialize the recommendation system
                recommender = ProductRecommendationSystem(spark)
                
                # Try to load existing model
                try:
                    model_loaded = recommender.load_model(
                        "models/recommendation_pipeline_hope", 
                        "data/vectorized_products_hope"
                    )
                    
                    if model_loaded:
                        st.session_state.recommender_system = recommender
                        st.success("Recommendation system loaded successfully!")
                        return recommender
                    else:
                        st.warning("Could not load pre-trained model")
                        st.info("Please train the model first using the PySpark script")
                        st.session_state.recommender_system = None
                        return None
                        
                except Exception as load_error:
                    st.warning(f"Could not load pre-trained model: {load_error}")
                    st.info("Please train the model first using the PySpark script")
                    st.session_state.recommender_system = None
                    return None
        else:
            return st.session_state.recommender_system
            
    except Exception as e:
        st.error(f"Error initializing recommendation system: {e}")
        st.session_state.recommender_system = None
        return None

def get_ingredient_recommendations_ui(ingredients: List[Dict], filters: Dict):
    """UI function to get and display recommendations for ingredients"""
    if not ingredients:
        return
    
    # Initialize recommendation system
    recommender = initialize_recommendation_system()
    
    if recommender is None:
        st.warning("Recommendation system not available. Please ensure the model is trained and saved.")
        return
    
    st.markdown("---")
    st.markdown("## 🛍️ Product Recommendations")
    
    # Select ingredient for recommendations
    ingredient_names = [ing['name'] for ing in ingredients]
    selected_ingredient = st.selectbox(
        "Select an ingredient to get product recommendations:",
        ingredient_names,
        key="ingredient_selector"
    )
    
    if st.button("Get Product Recommendations", key="get_recs_btn"):
        with st.spinner(f"Finding products for {selected_ingredient}..."):
            # Get raw recommendations
            raw_recommendations = get_product_recommendations(
                selected_ingredient, 
                recommender, 
                method="cosine", 
                num_recommendations=10
            )
            
            # Apply filters
            filtered_recommendations = apply_product_filters(raw_recommendations, filters)
            
            # Display results
            if filtered_recommendations:
                display_product_recommendations(selected_ingredient, filtered_recommendations)
                
                # Show statistics
                st.markdown("### 📊 Recommendation Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Found", len(raw_recommendations))
                with col2:
                    st.metric("After Filters", len(filtered_recommendations))
            else:
                st.info(f"No products found for '{selected_ingredient}' matching your filters. Try adjusting the filters.")

def display_enhanced_ingredient_analysis(ingredients: List[Dict]):
    """Enhanced ingredient analysis with more details"""
    if not ingredients:
        return
    
    st.markdown("### 📊 Enhanced Ingredient Analysis")
    
    # Category distribution
    categories = {}
    allergen_count = {}
    
    for ing in ingredients:
        cat = ing['category']
        categories[cat] = categories.get(cat, 0) + 1
        
        for allergen in ing['allergens']:
            allergen_count[allergen] = allergen_count.get(allergen, 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Category Distribution")
        for category, count in sorted(categories.items()):
            percentage = (count / len(ingredients)) * 100
            st.write(f"**{category.title()}:** {count} ({percentage:.1f}%)")
    
    with col2:
        st.markdown("#### ⚠️ Allergen Analysis")
        if allergen_count:
            for allergen, count in sorted(allergen_count.items()):
                percentage = (count / len(ingredients)) * 100
                st.write(f"**{allergen.title()}:** {count} ingredients ({percentage:.1f}%)")
        else:
            st.write("No common allergens detected")

def export_ingredients_data(ingredients: List[Dict], recipe_text: str):
    """Export ingredients data as JSON"""
    if not ingredients:
        return
    
    export_data = {
        'recipe_text': recipe_text,
        'extraction_date': str(pd.Timestamp.now()),
        'total_ingredients': len(ingredients),
        'ingredients': ingredients
    }
    
    json_str = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="📥 Download Ingredients Data (JSON)",
        data=json_str,
        file_name=f"ingredients_extract_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    
# Main header
st.markdown("""
<div class="main-header">
    <h1>🍽️ Food Recommendation System 🍽️</h1>
    <p>Select your country, enter a recipe, and discover ingredients with smart filtering</p>
</div>
""", unsafe_allow_html=True)

# Country Selection
st.markdown("## 🌍 Select Your Country")
col1, col2, col3, col4 = st.columns(4)

country_names = list(COUNTRIES.keys())
cols = [col1, col2, col3, col4]

for i, (country, data) in enumerate(COUNTRIES.items()):
    with cols[i]:
        selected_class = "selected" if st.session_state.selected_country == i else ""
            
        # Display the visual card with flag image
        country_card_html = f"""
        <div class="country-card {selected_class}">
            <img src="{data['flag_url']}" class="country-flag" alt="{country} flag">
            <div class="country-info">
                <h3 style="margin: 5px 0; color: #2d3748; font-size: 1.2rem;">{country}</h3>
                <p style="margin: 0; color: #4a5568; font-size: 0.9rem;">{data['cuisine']} Cuisine</p>
            </div>
        </div>
        """
        
        st.markdown(country_card_html, unsafe_allow_html=True)
        
        # Clickable button below the card
        if st.button(
            f"Select {country}",
            key=f"country_{i}",
            help=f"Select {country}",
            use_container_width=True
        ):
            st.session_state.selected_country = i
            st.rerun()

# Display selected country info
selected_country_name = country_names[st.session_state.selected_country]
selected_country_data = COUNTRIES[selected_country_name]

st.info(f"Your Selected Country IS : **{selected_country_name}**")

st.markdown("---")

# Recipe Input Form
st.markdown("## 📝 Enter Your Recipe")

form = st.form(key="recipe_form")
with form:
    recipe_text = st.text_area(
        "Paste your recipe here:",
        value=st.session_state.recipe_text,
        height=200,
        placeholder="Enter your recipe... For example:\n\nCook spaghetti in boiling salted water. In a separate pan, heat olive oil and sauté chopped onion and minced garlic until soft. Add ground beef and cook until browned.",
        help="Enter a detailed recipe"
    )
    
    col1 = st.columns(1)[0]
    
    st.markdown("""
        <style>
        /* Target any button inside a form inside a column */
        div[data-testid="column"] form button {
            width: 100%;
            background: linear-gradient(135deg, #DFD0B8 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
            padding: 0.75em 1em !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            transition: 0.3s ease-in-out;
        }
        div[data-testid="column"] form button:hover {
            opacity: 0.9;
            transform: scale(1.01);
        }
        </style>
    """, unsafe_allow_html=True)

    # Only the extract button
    with col1:
        extract_button = st.form_submit_button(
            "🔍 Extract Ingredients",
            use_container_width=True
        )
        
# Process form submission
if extract_button and recipe_text.strip():
    st.session_state.recipe_text = recipe_text
    with st.spinner("Extracting ingredients..."):
        st.session_state.ingredients = extract_ingredients_advanced(recipe_text)
    st.session_state.show_filters = True
    st.rerun()

# Display results
if st.session_state.ingredients:
    st.markdown("---")
    
    # Create layout with sidebar for filters
    if st.session_state.show_filters:
        col_main, col_sidebar = st.columns([2, 1])
    else:
        col_main = st.container()
        col_sidebar = None
    
    with col_main:
        st.markdown("## 🥘 Extracted Ingredients")
        
        # Apply filters if sidebar is active
        display_ingredients = st.session_state.ingredients
        
        if col_sidebar is not None:
            # Get filter values from sidebar
            with col_sidebar:
                st.markdown("## 🔽 Filter Criteria")
                
                # Dietary restrictions
                st.markdown("### 🌱 Dietary Restrictions")
                dietary_filters = []
                if st.checkbox("Vegetarian", key="veg"):
                    dietary_filters.append("vegetarian")
                if st.checkbox("Vegan", key="vegan"):
                    dietary_filters.append("vegan")
                if st.checkbox("Gluten-Free", key="gluten_free"):
                    dietary_filters.append("gluten-free")
                if st.checkbox("Dairy-Free", key="dairy_free"):
                    dietary_filters.append("dairy-free")
                if st.checkbox("Nut-Free", key="nut_free"):
                    dietary_filters.append("nut-free")
                if st.checkbox("Egg-Free", key="egg_free"):
                    dietary_filters.append("egg-free")
                if st.checkbox("Peanuts-Free", key="peanuts_free"):
                    dietary_filters.append("peanuts-free")
                if st.checkbox("Wheat-Free", key="wheat_free"):
                    dietary_filters.append("wheat-free")
                if st.checkbox("Fish-Free", key="fish_free"):
                    dietary_filters.append("fish-free")
                
                # NutriScore
                st.markdown("### 🍳 NutriScore")
                cuisine_style = st.selectbox(
                    "NutriScore",
                    ["A", "B", "C", "D","E"],
                    key="cuisine_style"
                )
                
                # EcoScore
                st.markdown("### 🌱 EcoScore")
                meal_type = st.selectbox(
                    "EcoScore",
                    ["a", "b", "c", "d", "e"],
                    key="meal_type"
                )
                
                # Clear filters
                if st.button("🔄 Clear All Filters", use_container_width=True):
                    st.rerun()
                
                # Apply filters
                filters = {'dietary': dietary_filters}
                display_ingredients = filter_ingredients(st.session_state.ingredients, filters)
        
        # Display ingredients
        if display_ingredients:
            st.success(f"Found {len(display_ingredients)} ingredients matching your criteria!")
            
            # Create ingredient grid
            num_cols = 3
            ingredient_cols = st.columns(num_cols)
            
            tet=[]
            for ingredient in display_ingredients:
                if ingredient['name'] in VALID_INGREDIENTS_FOR_RECOMMENDATIONS:
                    tet.append(ingredient)
                
            for i, ingredient in enumerate(tet):
                col_idx = i % num_cols
                with ingredient_cols[col_idx]:
                    # Create allergen text
                    allergen_text = f"Allergens: {', '.join(ingredient['allergens'])}" if ingredient['allergens'] else "No known allergens"
                    
                    st.markdown(f"""
                    <div class="ingredient-card">
                        <h4>🥄 {ingredient['name'].title()}</h4>
                        <p><strong>Category:</strong> {ingredient['category'].title()}</p>
                        <p><small>{allergen_text}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No ingredients match your current filter criteria. Try adjusting your filters.")
            
        # Display rec
        st.markdown("## 🔮 Get Smart Product Recommendations")
        if st.button("🚀 Recommend Based on Ingredients", key="recommend_button"):
            with st.spinner("Generating recommendations..."):
                recommender = initialize_recommendation_system()
                if recommender:
                    # Make sure display_ingredients is defined
                    if 'display_ingredients' in locals() or 'display_ingredients' in globals():
                        for ingredient in display_ingredients[:5]:  # Limit to 5 ingredients max
                            try:
                                recs = recommender.get_filtered_recommendations(
                                    product_name=ingredient['name'],
                                    categories=None,
                                    filters={
                                        'gluten_free': 'gluten-free' in dietary_filters if 'dietary_filters' in locals() else False,
                                        'dairy_free': 'dairy-free' in dietary_filters if 'dietary_filters' in locals() else False,
                                        'nutriscore': cuisine_style if 'cuisine_style' in locals() else None,
                                        'ecoscore': meal_type if 'meal_type' in locals() else None
                                    },
                                    num_recommendations=6
                                )
                                if recs:
                                    ingredient_lower = ingredient['name'].lower().strip()
                                    if ingredient_lower in VALID_INGREDIENTS_FOR_RECOMMENDATIONS: 
                                        display_product_recommendations(ingredient['name'], recs)
                                    else:
                                        continue
                                else:
                                    st.info(f"No recommendations found for {ingredient['name']}.")
                            except Exception as e:
                                st.error(f"Error getting recommendations for {ingredient['name']}: {str(e)}")
                    else:
                        st.error("No ingredients found. Please analyze an image first.")
                else:
                    st.error("Recommendation system not initialized properly.")
    
    # Additional features
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Ingredient Analysis")
        if display_ingredients:
            categories = {}
            for ing in display_ingredients:
                cat = ing['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            for category, count in categories.items():
                st.write(f"**{category.title()}:** {count} ingredients")
    
    with col2:
        st.markdown("### 🎯 Recipe Suggestions")
        country_specialties = COUNTRIES[selected_country_name]['specialties']
        st.write(f"Popular in {selected_country_name}:")
        for specialty in country_specialties:
            st.write(f"• {specialty.title()}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 30px;">
    <p>🍽️ Food Recommendation System | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)



