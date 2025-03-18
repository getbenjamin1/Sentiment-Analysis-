import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

# Load the processed dataset
df = pd.read_csv("processed_data_labeled.csv")

# --- Convert numeric labels to integer and then map to text ---
# Adjust the mapping to match your own label definitions
df['label'] = df['label'].round().astype(int)  # Ensure no decimals
label_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
df['label_text'] = df['label'].map(label_mapping)

# Extract unique data sources for the dropdown
data_sources = df['source'].unique().tolist()
# Add an 'All' option at the front for showing the entire dataset
data_sources.insert(0, 'All')

# Initialise the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Interactive Sentiment Dashboard"),
    
    # Dropdown for selecting a data source or all
    dcc.Dropdown(
        id='source-dropdown',
        options=[{'label': source, 'value': source} for source in data_sources],
        value='All',  # Default to showing all sources
        clearable=False
    ),
    
    # Graph component to display the bar chart
    dcc.Graph(id='sentiment-bar')
])

# Define the callback to update the bar chart based on the selected data source
@app.callback(
    Output('sentiment-bar', 'figure'),
    [Input('source-dropdown', 'value')]
)
def update_bar_chart(selected_source):
    # If 'All' is selected, don't filter
    if selected_source == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['source'] == selected_source]

    # Group by 'label_text' and count how many entries fall under each label
    sentiment_counts = filtered_df.groupby('label_text').size().reset_index(name='count')

    # Create a bar chart using the grouped data
    fig = px.bar(
        sentiment_counts,
        x='label_text',     # x-axis: sentiment label text
        y='count',          # y-axis: number of entries for that label
        color='label_text', # colour the bars by label text
        title=f"Sentiment Distribution for {selected_source}"
    )

    #Update layout for a cleaner look
    fig.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Count",
        legend_title="Sentiment"
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
