import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

# Load the processed dataset (labelled data)
df = pd.read_csv("processed_data_labeled.csv")

# Convert numeric labels to integer and map to text
df['label'] = df['label'].round().astype(int)
label_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
df['label_text'] = df['label'].map(label_mapping)

# Extract unique data sources for the dropdown from the labelled data,
# filtering out any "Twitter" entries to avoid duplication.
data_sources = df['source'].unique().tolist()
data_sources = [src for src in data_sources if src.lower() != 'twitter']

# Insert an 'All' option at the beginning and then append two Twitter options.
data_sources.insert(0, 'All')
data_sources.append('Twitter Over Time')
data_sources.append('Twitter Distribution')

# -------------------------------------------
# Prepare Twitter sentiment data
# -------------------------------------------
df_twitter = pd.read_csv('twitter_data_with_sentiment_updated.csv')
df_twitter['created_at'] = pd.to_datetime(df_twitter['created_at'])
# Create a 'date' column by extracting the date part (as datetime) from 'created_at'
df_twitter['date'] = pd.to_datetime(df_twitter['created_at'].dt.date)

# Prepare data for Twitter Sentiment Over Time (line chart)
twitter_sentiment_over_time = df_twitter.groupby(['date', 'sentiment']).size().reset_index(name='count')
fig_twitter = px.line(
    twitter_sentiment_over_time,
    x='date',
    y='count',
    color='sentiment',
    markers=True,
    title="Twitter Sentiment Over Time"
)
fig_twitter.update_layout(
    xaxis_title="Date",
    yaxis_title="Number of Tweets",
    legend_title="Sentiment"
)

# Prepare data for Twitter Sentiment Distribution (bar chart)
twitter_sentiment_distribution = df_twitter.groupby('sentiment').size().reset_index(name='count')
fig_twitter_distribution = px.bar(
    twitter_sentiment_distribution,
    x='sentiment',
    y='count',
    color='sentiment',
    title="Twitter Sentiment Distribution"
)
fig_twitter_distribution.update_layout(
    xaxis_title="Sentiment",
    yaxis_title="Count",
    legend_title="Sentiment"
)

# Initialise the Dash app
app = dash.Dash(__name__)

# Define the layout with one graph that updates based on the dropdown selection
app.layout = html.Div([
    html.H1("Interactive Sentiment Dashboard"),
    
    dcc.Dropdown(
        id='source-dropdown',
        options=[{'label': source, 'value': source} for source in data_sources],
        value='All',  # Default to "All"
        clearable=False
    ),
    
    dcc.Graph(id='sentiment-graph')
])

# Callback to update the graph based on the dropdown selection
@app.callback(
    Output('sentiment-graph', 'figure'),
    [Input('source-dropdown', 'value')]
)
def update_graph(selected_source):
    if selected_source == "Twitter Over Time":
        return fig_twitter
    elif selected_source == "Twitter Distribution":
        return fig_twitter_distribution
    else:
        # For "All" or any specific labelled data source, filter accordingly.
        if selected_source == 'All':
            filtered_df = df
        else:
            filtered_df = df[df['source'] == selected_source]
        
        # Group by sentiment label and count the number of entries.
        sentiment_counts = filtered_df.groupby('label_text').size().reset_index(name='count')
        
        # Create a bar chart for the sentiment distribution.
        fig = px.bar(
            sentiment_counts,
            x='label_text',
            y='count',
            color='label_text',
            title=f"Sentiment Distribution for {selected_source}"
        )
        fig.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Count",
            legend_title="Sentiment"
        )
        return fig

if __name__ == '__main__':
    app.run_server(debug=True)
