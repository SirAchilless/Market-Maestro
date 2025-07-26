import streamlit as st
import json
import os
from datetime import datetime

# Emoji mood categories with predefined options
EMOJI_MOODS = {
    'Bullish': ['🚀', '📈', '💰', '🟢', '⬆️', '🔥', '💎', '🌟'],
    'Bearish': ['📉', '🔴', '⬇️', '💸', '😰', '🆘', '⚠️', '🩸'],
    'Neutral': ['➡️', '😐', '🤔', '⚖️', '🔵', '⏸️', '🤷', '📊'],
    'Watchful': ['👀', '🔍', '🎯', '📱', '⏰', '🔔', '👁️', '🕵️'],
    'Excited': ['🎉', '💪', '🔝', '✨', '🎊', '🌈', '🦾', '🏆'],
    'Cautious': ['⚠️', '🚧', '🛑', '⛔', '🔶', '📛', '🚨', '⏳']
}

def get_mood_file_path():
    """Get the path to the emoji mood storage file"""
    os.makedirs("user_data", exist_ok=True)
    return "user_data/emoji_moods.json"

def load_user_emoji_moods():
    """Load user's custom emoji moods for stocks"""
    try:
        mood_file = get_mood_file_path()
        if os.path.exists(mood_file):
            with open(mood_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Error loading emoji moods: {str(e)}")
    return {}

def save_user_emoji_moods(moods):
    """Save user's emoji moods to file"""
    try:
        mood_file = get_mood_file_path()
        with open(mood_file, 'w') as f:
            json.dump(moods, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving emoji moods: {str(e)}")
        return False

def get_stock_emoji(symbol):
    """Get the emoji mood for a specific stock"""
    moods = load_user_emoji_moods()
    return moods.get(symbol, {}).get('emoji', '📊')  # Default to chart emoji

def get_stock_mood_category(symbol):
    """Get the mood category for a specific stock"""
    moods = load_user_emoji_moods()
    return moods.get(symbol, {}).get('category', 'Neutral')

def set_stock_emoji_mood(symbol, emoji, category):
    """Set emoji mood for a stock"""
    moods = load_user_emoji_moods()
    moods[symbol] = {
        'emoji': emoji,
        'category': category,
        'updated_at': datetime.now().isoformat()
    }
    return save_user_emoji_moods(moods)

def remove_stock_emoji_mood(symbol):
    """Remove emoji mood for a stock"""
    moods = load_user_emoji_moods()
    if symbol in moods:
        del moods[symbol]
        return save_user_emoji_moods(moods)
    return True

def display_emoji_mood_selector(symbol, key_suffix=""):
    """Display emoji mood selector widget for a stock"""
    st.markdown(f"### {get_stock_emoji(symbol)} Emoji Mood for {symbol}")
    
    # Get current mood
    current_emoji = get_stock_emoji(symbol)
    current_category = get_stock_mood_category(symbol)
    
    # Category selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox(
            "Mood Category",
            options=list(EMOJI_MOODS.keys()),
            index=list(EMOJI_MOODS.keys()).index(current_category),
            key=f"mood_category_{symbol}_{key_suffix}"
        )
    
    with col2:
        # Display current emoji
        st.markdown(f"**Current:** {current_emoji}")
    
    # Emoji selector for the selected category
    st.markdown(f"**Choose {selected_category} Emoji:**")
    
    # Display emojis in a grid
    emoji_cols = st.columns(4)
    selected_emoji = current_emoji
    
    for i, emoji in enumerate(EMOJI_MOODS[selected_category]):
        col_idx = i % 4
        with emoji_cols[col_idx]:
            if st.button(
                emoji, 
                key=f"emoji_{symbol}_{emoji}_{key_suffix}",
                help=f"Set {emoji} for {symbol}",
                use_container_width=True
            ):
                selected_emoji = emoji
                # Save the selection
                if set_stock_emoji_mood(symbol, emoji, selected_category):
                    st.success(f"Updated {symbol} mood to {emoji}")
                    st.rerun()
    
    # Additional options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(f"🔄 Reset {symbol}", key=f"reset_{symbol}_{key_suffix}"):
            if remove_stock_emoji_mood(symbol):
                st.success(f"Reset {symbol} to default mood")
                st.rerun()
    
    with col2:
        # Custom emoji input
        custom_emoji = st.text_input(
            "Custom Emoji", 
            placeholder="Type any emoji",
            key=f"custom_{symbol}_{key_suffix}",
            max_chars=2
        )
        
    with col3:
        if custom_emoji and st.button(f"✅ Use Custom", key=f"save_custom_{symbol}_{key_suffix}"):
            if set_stock_emoji_mood(symbol, custom_emoji, selected_category):
                st.success(f"Set custom emoji {custom_emoji} for {symbol}")
                st.rerun()

def display_mood_stats():
    """Display statistics about user's emoji moods"""
    moods = load_user_emoji_moods()
    
    if not moods:
        st.info("No emoji moods set yet. Start by setting moods for your favorite stocks!")
        return
    
    st.markdown("### 📊 Your Emoji Mood Statistics")
    
    # Category distribution
    category_counts = {}
    for stock_data in moods.values():
        category = stock_data.get('category', 'Neutral')
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Display stats
    cols = st.columns(len(category_counts))
    for i, (category, count) in enumerate(category_counts.items()):
        with cols[i]:
            # Get a representative emoji for the category
            repr_emoji = EMOJI_MOODS[category][0] if category in EMOJI_MOODS else '📊'
            st.metric(f"{repr_emoji} {category}", count)
    
    # Recent updates
    if moods:
        st.markdown("### 🕒 Recently Updated")
        sorted_moods = sorted(
            moods.items(), 
            key=lambda x: x[1].get('updated_at', ''), 
            reverse=True
        )
        
        for symbol, data in sorted_moods[:5]:
            emoji = data.get('emoji', '📊')
            category = data.get('category', 'Neutral')
            updated = data.get('updated_at', 'Unknown')[:10]  # Show date only
            st.write(f"{emoji} **{symbol}** - {category} *(Updated: {updated})*")

def get_stocks_by_mood_category(category):
    """Get all stocks with a specific mood category"""
    moods = load_user_emoji_moods()
    stocks = []
    
    for symbol, data in moods.items():
        if data.get('category') == category:
            stocks.append({
                'symbol': symbol,
                'emoji': data.get('emoji', '📊'),
                'updated_at': data.get('updated_at', '')
            })
    
    return sorted(stocks, key=lambda x: x['updated_at'], reverse=True)

def display_mood_categories_overview():
    """Display overview of all mood categories with stocks"""
    moods = load_user_emoji_moods()
    
    if not moods:
        return
    
    st.markdown("### 🎭 Mood Categories Overview")
    
    tabs = st.tabs(list(EMOJI_MOODS.keys()))
    
    for i, category in enumerate(EMOJI_MOODS.keys()):
        with tabs[i]:
            stocks = get_stocks_by_mood_category(category)
            
            if stocks:
                st.markdown(f"**{len(stocks)} stocks in {category} mood:**")
                
                # Display in grid
                if len(stocks) > 0:
                    cols = st.columns(min(4, len(stocks)))
                    for j, stock in enumerate(stocks):
                        col_idx = j % 4
                        with cols[col_idx]:
                            st.markdown(f"""
                            <div style='text-align: center; padding: 10px; margin: 5px; 
                                        border-radius: 8px; background-color: #f0f0f0;'>
                                <div style='font-size: 24px;'>{stock['emoji']}</div>
                                <div style='font-weight: bold;'>{stock['symbol']}</div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info(f"No stocks set to {category} mood yet.")

def create_emoji_mood_page():
    """Create a complete emoji mood management page"""
    st.title("🎭 Personalized Stock Emoji Mood Selector")
    st.markdown("Set custom emoji moods for your stocks to personalize your trading experience!")
    
    # Display current mood stats
    display_mood_stats()
    
    st.markdown("---")
    
    # Stock selector for setting mood
    st.markdown("### 🎯 Set Emoji Mood for Stock")
    
    # Import stock lists for selection
    try:
        from utils.stock_lists import NIFTY_50, NIFTY_100
        available_stocks = sorted(set(NIFTY_50 + NIFTY_100))
    except:
        available_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    
    # Allow manual input as well
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_stock = st.selectbox(
            "Choose Stock",
            options=available_stocks,
            index=0
        )
    
    with col2:
        manual_stock = st.text_input(
            "Or type symbol",
            placeholder="e.g., WIPRO"
        )
    
    # Use manual input if provided
    stock_to_set = manual_stock.upper() if manual_stock else selected_stock
    
    if stock_to_set:
        st.markdown("---")
        display_emoji_mood_selector(stock_to_set, "main_page")
    
    st.markdown("---")
    
    # Display mood categories overview
    display_mood_categories_overview()