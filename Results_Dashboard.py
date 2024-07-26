import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import scipy as sp

from scipy import stats
from scipy.stats import mannwhitneyu, norm
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import cdist

# Function to extract abbreviation from text in brackets
def extract_abbreviation(text):
    start = text.find('(')
    end = text.find(')')
    if start != -1 and end != -1 and start < end:
        return text[start+1:end]
    return text

st.set_page_config(page_title="SCALE Analysis",page_icon=":bar_chart:",layout="wide")

st.title(":bar_chart: SCALE Analysis")

df_pre = pd.read_excel("PRE_RAW_COMPLETED_240711.xlsx")
df_post = pd.read_excel("POST_RAW_COMPLETED_240711.xlsx")

# Extract abbreviations for "College" and "School" columns
df_pre['College'] = df_pre['College'].apply(extract_abbreviation)
df_pre['School'] = df_pre['School'].apply(extract_abbreviation)

tab_names = [":male-technologist: **Data**", ":large_green_circle: **Pre-Intervetion**", ":large_orange_circle: **Post-Intervention**", ":bar_chart: **Analysis**"]

tabs = st.tabs(tab_names)

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

with tabs[0]:
    st.header(":green[PRE-Intervention Raw Data]")
    st.dataframe(df_pre)
    
    st.header(":green[POST-Intervention Raw Data]")
    st.dataframe(df_post)

with tabs[1]:
    st.subheader("Filter by Discipline or College:")
    # Create a unique list of colleges and disciplines for the selectbox
    unique_colleges = df_pre['College'].unique().tolist()
    unique_disciplines = df_pre['Discipline'].unique().tolist()
    
    # Add a selectbox for filtering by college or discipline
    selected_filter = st.selectbox(
        "",
        options=["All"] + unique_colleges + unique_disciplines
    )
    
    # Filter dataframe based on the selected filter
    if selected_filter != "All":
        if selected_filter in unique_colleges:
            df_pre = df_pre[df_pre['College'] == selected_filter]
        elif selected_filter in unique_disciplines:
            df_pre = df_pre[df_pre['Discipline'] == selected_filter]

    total_entries = len(df_pre)
    st.header(":green[Participant Demographic]")
    col1_1, col1_2, col1_3 = st.columns(3)
    
    with col1_1:
        ##### COLLEGE #####
        st.subheader(":blue[College:]")
        counts_college = df_pre['College'].value_counts().reset_index()
        counts_college.columns = ['College', 'Count']
        fig_college = px.pie(counts_college, values='Count', names='College')
        fig_college.update_layout(legend=dict(x=0.1,y=1,traceorder='normal'))
        st.plotly_chart(fig_college,use_container_width=True)
    
    with col1_2:
        ##### SCHOOL #####
        st.subheader(":blue[School:]")
        counts_school = df_pre['School'].value_counts().reset_index()
        counts_school.columns = ['School', 'Count']
        fig_school = px.pie(counts_school, values='Count', names='School')
        fig_school.update_layout(legend=dict(x=0.0,y=1,traceorder='normal'))
        st.plotly_chart(fig_school,use_container_width=True)
        
    with col1_3:
        ##### YEAR OF STUDY #####
        st.subheader(":blue[Year of Study:]")
        counts_study = df_pre['Year'].value_counts().reset_index()
        counts_study.columns = ['Year', 'Count']
        fig_study = px.pie(counts_study, values='Count', names='Year')
        fig_study.update_layout(legend=dict(x=0.75,y=1,traceorder='normal'))
        st.plotly_chart(fig_study,use_container_width=True)
    
    st.header(":green[Response to Survey Questions]")
    
    col2_1, col2_2 = st.columns((2,1))
    
    with col2_1:
        st.subheader(":blue[Non-Core Course Selection:]")
        st.markdown("This question is about the degree of importance of these factors when choosing a non-core course (MPE, BDE).")

        # List of specific columns to be plotted
        Question_4 = [
        '2.1 (Q4_A_14)', '2.1 (Q4_A_13)', '2.1 (Q4_A_12)', '2.1 (Q4_A_11)',
        '2.1 (Q4_A_10)', '2.1 (Q4_A_9)', '2.1 (Q4_A_8)', '2.1 (Q4_A_7)',
        '2.1 (Q4_A_6)', '2.1 (Q4_A_5)', '2.1 (Q4_A_4)', '2.1 (Q4_A_3)',
        '2.1 (Q4_A_2)', '2.1 (Q4_A_1)']
        
        # Create a mapping from original column names to new labels ("A", "B", "C", etc.)
        new_labels = [chr(65 + i) for i in range(len(Question_4))]
        column_label_mapping = dict(zip(Question_4, new_labels[::-1]))
        
        # Convert numerical responses to Likert scale labels
        likert_labels = {1: 'Not important at all', 2: 'Somewhat unimportant', 3: 'Somewhat important', 4: 'Very important'}
        
        # Initialize a DataFrame to store the frequency counts
        frequency_data_4 = {'Question': [], 'Not important at all': [], 'Somewhat unimportant': [], 'Somewhat important': [], 'Very important': []}
        hover_data_4 = {'Question': [], 'Not important at all': [], 'Somewhat unimportant': [], 'Somewhat important': [], 'Very important': []}
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_4:
            frequency_data_4['Question'].append(column_label_mapping[question])
            hover_data_4['Question'].append(column_label_mapping[question])
            counts = df_pre[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_4[label].append(percentage)
                hover_data_4[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_pre_4 = pd.DataFrame(frequency_data_4)
        hover_df_pre_4 = pd.DataFrame(hover_data_4)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_pre_4['Question'],
                x=frequency_df_pre_4[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_pre_4[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_pre_4['Question']) - 0.5,
            line=dict(
                color='Black',
                width=2,
                dash='dash',
            ),
        )
        
        fig_diverging.update_layout(
            barmode='relative',
            xaxis_title='Percentage (%)',
            yaxis_title='Questions',
            legend_title='Responses',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            width=1100,  # Set the width of the figure
            height=500,   # Set the height of the figure
            legend=dict(traceorder='reversed')
            )
        
        st.plotly_chart(fig_diverging,use_container_width=True)
    
    with col2_2:
        # List of specific columns to be plotted
        Question_4_Description = [
            'Relevance to your major or desired career field',
            'Recommendation from a friend or peer',
            'Recommendation from a career/course advisor',
            'Instructor\'s reputation or teaching style',
            'Difficulty level (academic content) or course workload',
            'Timing of the course (fits well in your schedule)',
            'Opportunity to develop specific skills or competencies',
            'Alignment with personal interests or hobbies',
            'Course reviews or ratings from other students',
            'Potential to boost your overall GPA',
            'Availability of resources (like textbooks, online resources, etc.)',
            'Class size (preference for smaller or larger classes)',
            'Mode of delivery (in-person, online, hybrid)',
            'Fulfilment for minor or secondary academic discipline'
            ]
        
        mapping_table = pd.DataFrame({
        'Label': new_labels,
        'Survey Question': Question_4_Description
        })
        
        st.dataframe(mapping_table,height=(len(column_label_mapping)+1)*35+3,use_container_width=True,hide_index=True)
    
    col3_1, col3_2, col3_3 = st.columns((1,3,1))
    with col3_2:
        # Extract the 'RecordID' column for labeling
        record_ids = '...' + df_pre['Record ID'].astype(str).str[-4:]
    
        # Drop the 'RecordID' column for clustering analysis
        df_pre2 = df_pre[Question_4]
        
        # Relabel columns to A-N
        new_column_labels = [chr(i) for i in range(ord('A'), ord('A') + len(Question_4))]

        df_pre2.columns = new_column_labels
        
        # Standardize the data
        #scaler = StandardScaler()
        #standardized_data = scaler.fit_transform(df_pre2)  # Replace with relevant columns
    
        # Define a custom colormap with 4 colors
        custom_cmap = ListedColormap(['#de425b', '#f3babc', '#aecdc2', '#488f31'])
    
        # Transpose the data to have questions as rows and responses as columns
        #transposed_data = standardized_data.T

        # Plot the clustermap with boxes
        cluster_map = sns.clustermap(df_pre2.T, method='ward', metric='euclidean', cmap=custom_cmap, figsize=(20, 10), 
                                     dendrogram_ratio=(.1, .1), cbar_pos=None, 
                                     linewidths=2.0, linecolor='black')
        
        cluster_map.ax_heatmap.set_xticks(np.arange(len(record_ids))+0.5)
        cluster_map.ax_heatmap.set_xticklabels(record_ids, rotation=45, ha='right', fontsize=6)
    
        # Adjust y-tick labels: control the shift up
        for label in cluster_map.ax_heatmap.yaxis.get_majorticklabels():
            label.set_verticalalignment('center')
            label.set_position((label.get_position()[0], label.get_position()[1] + 0.5))
            label.set_x(-0.015)
        
        plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.5)
        st.pyplot(cluster_map.fig,use_container_width=True)
        
    col4_1, col4_2 = st.columns((20,1))
    
    with col4_1:
        st.subheader(":blue[Industry and Skill Requirements:]")
        st.markdown("Which industry is the FIRST (1st) choice for your first job after graduation?")
        
        ##### INDUSTRY #####
        # Columns to plot
        columns = ['Discipline', 'College', 'School', 'Programme', 'Industry']
        #columns = ['School', 'Programme', 'Industry']
        
        # Count occurrences of each combination of "College", "School", "Programme", and "Industry"
        df_pre_counts = df_pre.groupby(columns).size().reset_index(name='count')
        
        # Prepare data for the Sankey diagram
        all_nodes = list(pd.concat([df_pre[col] for col in columns]).unique())
        node_indices = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Create the Sankey diagram links
        links = []
        for _, row in df_pre_counts.iterrows():
            for i in range(len(columns) - 1):
                source = row[columns[i]]
                target = row[columns[i + 1]]
                value = row['count']
                links.append({
                    'source': node_indices[source],
                    'target': node_indices[target],
                    'value': value
                })
        
        # Create the Sankey diagram nodes
        nodes = [{'name': node} for node in all_nodes]
        
        # Calculate incoming flow for each node for hover info
        node_incoming_flow = [0] * len(nodes)
        for link in links:
            node_incoming_flow[link['target']] += link['value']
        
        # Create the Sankey diagram
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[node['name'] for node in nodes],
                customdata=node_incoming_flow,
                hovertemplate='%{label}: %{customdata} incoming'
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                hoverinfo='skip'  # Skip hover info for links
            )
        ))
        
        fig.update_layout(
            font_size=12,
            width=1600,  # Set the width of the figure
            height=800   # Set the height of the figure
        )
        
        st.plotly_chart(fig,use_container_width=True)
    
    
    # Get unique values for the filter column
    filter_values = df_pre['Industry'].unique()
    
    # Multiselect for selecting filter values
    selected_values = st.multiselect("Filter by Industry", filter_values, default=filter_values)
    
    # Define stopwords
    stopwords = set(STOPWORDS)
    stopwords.add("skill")
    stopwords.add("skills")
    
    col5_1, col5_2 = st.columns(2)
    
    with col5_1:
        st.markdown("For my desired first choice, the skills that are important (in my opinion) are:")
        if selected_values:
            # Filter the DataFrame based on the selected values
            filtered_df_pre = df_pre[df_pre['Industry'].isin(selected_values)]
    
            if not filtered_df_pre.empty:
                # Combine all text from the three specific columns into one single string
                columns = ['3.2 (Q10_1)', '3.2 (Q10_2)', '3.2 (Q10_3)']
                combined_text = " ".join(filtered_df_pre[columns[0]].astype(str)).lower() + " " + " ".join(filtered_df_pre[columns[1]].astype(str)).lower() + " " + " ".join(filtered_df_pre[columns[2]].astype(str)).lower()
    
                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(combined_text)
    
                # Display the generated word cloud image
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt,use_container_width=True)
        
    with col5_2:
        st.markdown("The important skills that I currently lack are:")
        if selected_values:
            # Filter the DataFrame based on the selected values
            filtered_df_pre = df_pre[df_pre['Industry'].isin(selected_values)]
    
            if not filtered_df_pre.empty:
                # Combine all text from the three specific columns into one single string
                columns = ['3.4 (Q12_1)', '3.4 (Q12_2)', '3.4 (Q12_3)']
                combined_text = " ".join(filtered_df_pre[columns[0]].astype(str)).lower() + " " + " ".join(filtered_df_pre[columns[1]].astype(str)).lower() + " " + " ".join(filtered_df_pre[columns[2]].astype(str)).lower()
    
                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(combined_text)
    
                # Display the generated word cloud image
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt,use_container_width=True)
    
    col6_1, col6_2 = st.columns((2,1))
    
    with col6_1:
        st.markdown("From the skills deemed important for your first job, please rate your agreement with the following statements:")

        # List of specific columns to be plotted
        Question_11 = [
        '3.3 (Q11_A_10)', '3.3 (Q11_A_9)', '3.3 (Q11_A_8)', '3.3 (Q11_A_7)',
        '3.3 (Q11_A_6)', '3.3 (Q11_A_5)', '3.3 (Q11_A_4)', '3.3 (Q11_A_3)',
        '3.3 (Q11_A_2)', '3.3 (Q11_A_1)']
        
        # Create a mapping from original column names to new labels ("A", "B", "C", etc.)
        new_labels = [chr(65 + i) for i in range(len(Question_11))]
        column_label_mapping = dict(zip(Question_11, new_labels[::-1]))
        
        # Convert numerical responses to Likert scale labels
        likert_labels = {1: 'Strongly disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly agree'}
        
        # Initialize a DataFrame to store the frequency counts
        frequency_data_11 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        hover_data_11 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_11:
            frequency_data_11['Question'].append(column_label_mapping[question])
            hover_data_11['Question'].append(column_label_mapping[question])
            counts = df_pre[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_11[label].append(percentage)
                hover_data_11[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_pre_11 = pd.DataFrame(frequency_data_11)
        hover_df_pre_11 = pd.DataFrame(hover_data_11)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#f1f1f1', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_pre_11['Question'],
                x=frequency_df_pre_11[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_pre_11[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_pre_11['Question']) - 0.5,
            line=dict(
                color='Black',
                width=2,
                dash='dash',
            ),
        )
        
        fig_diverging.update_layout(
            barmode='relative',
            xaxis_title='Percentage (%)',
            yaxis_title='Questions',
            legend_title='Responses',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            width=1100,  # Set the width of the figure
            height=400,   # Set the height of the figure
            legend=dict(traceorder='reversed')
            )
        
        st.plotly_chart(fig_diverging,use_container_width=True)
        
    with col6_2:
        # List of specific columns to be plotted
        Question_11_Description = [
            'I have a good idea of the skills needed in my desired industry.',
            'I have developed these skills through the core and major prescribed elective (MPE) courses.',
            'I have developed these skills through the broadening and deeping elective (BDE) courses.',
            'I have developed these skills through the interdisciplinary collaborative core (ICC) courses.',
            'I have developed these skills through the university co-curricular activities.',
            'I have developed these skills outside the university.',
            'I feel ready to enter the job market (my desired industry) with the skills I currently possess.',
            'I believe I need to further develop the skills I listed to increase employability in my desired industry.',
            'I am aware of the skills I lack that are important in my desired industry.',
            'I have a clear plan to acquire or develop the skills I currently lack.'
            ]
        
        mapping_table = pd.DataFrame({
        'Label': new_labels,
        'Survey Question': Question_11_Description
        })
        
        st.dataframe(mapping_table,height=(len(column_label_mapping)+1)*35+3,use_container_width=True,hide_index=True)
        
    col7_1, col7_2 = st.columns((2,1))
    
    with col7_1:
        st.markdown("Please rate your agreement with the following statements:")

        # List of specific columns to be plotted
        Question_13 = ['3.6 (Q13_A_3)', '3.6 (Q13_A_2)', '3.6 (Q13_A_1)']
        
        # Create a mapping from original column names to new labels ("A", "B", "C", etc.)
        new_labels = [chr(65 + i) for i in range(len(Question_13))]
        column_label_mapping = dict(zip(Question_13, new_labels[::-1]))
        
        # Convert numerical responses to Likert scale labels
        likert_labels = {1: 'Strongly disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly agree'}
        
        # Initialize a DataFrame to store the frequency counts
        frequency_data_13 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        hover_data_13 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_13:
            frequency_data_13['Question'].append(column_label_mapping[question])
            hover_data_13['Question'].append(column_label_mapping[question])
            counts = df_pre[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_13[label].append(percentage)
                hover_data_13[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_pre_13 = pd.DataFrame(frequency_data_13)
        hover_df_pre_13 = pd.DataFrame(hover_data_13)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#f1f1f1', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_pre_13['Question'],
                x=frequency_df_pre_13[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_pre_13[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_pre_13['Question']) - 0.5,
            line=dict(
                color='Black',
                width=2,
                dash='dash',
            ),
        )
        
        fig_diverging.update_layout(
            barmode='relative',
            xaxis_title='Percentage (%)',
            yaxis_title='Questions',
            legend_title='Responses',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            width=1100,  # Set the width of the figure
            height=300,   # Set the height of the figure
            legend=dict(traceorder='reversed')
            )
        
        st.plotly_chart(fig_diverging,use_container_width=True)
        
    with col7_2:
        # List of specific columns to be plotted
        Question_13_Description = [
            'I know the resources the university has to help me acquire or develop the skills I currently lack.',
            'I know the targeted training and courses the university has to help me close my skills gap.',
            'I know the industry opportunities the university has to provide me with the relevant skills training.'
            ]
        
        mapping_table = pd.DataFrame({
        'Label': new_labels,
        'Survey Question': Question_13_Description
        })
        
        st.dataframe(mapping_table,height=(len(column_label_mapping)+1)*35+3,use_container_width=True,hide_index=True)

    col8_1, col8_2 = st.columns((2,1))
    
    with col8_1:
        st.markdown("Rank the following skill groups, in order of importance, according to how you perceive employers will rank them.")
        Distribution = [
            [34, 30, 28, 39, 57, 0],
            [12, 22, 58, 45, 29, 22],
            [14, 51, 27, 41, 35, 34],
            [0, 51, 41, 29, 33, 20]
        ]
        
        
        # Sample data with provided distribution
        data = {
            'Rank': [1, 2, 3, 4],
            'Choice': ['Soft', 'Functional', 'Requirement', 'Domain'],
            'Distribution': [
                [18.085106382978726, 15.957446808510639, 14.893617021276595, 20.74468085106383, 30.319148936170215, 0.0],
                [6.382978723404255, 11.702127659574469, 30.851063829787233, 23.93617021276596, 15.425531914893616, 11.702127659574469],
                [7.446808510638298, 27.127659574468083, 14.361702127659576, 21.808510638297875, 18.617021276595743, 10.638297872340425],
                [0.0, 27.127659574468083, 21.808510638297875, 15.425531914893616, 17.5531914893617, 18.085106382978726]
            ],
            'Score': [2.799, 2.526, 2.390, 2.286],
            'Times Ranked': [57, 29, 35, 33]
        }
        colors = ['#ffffff', '#de425b', '#f3babc', '#aecdc2', '#488f31', '#ffffff']
        
        # Create DataFrame
        df_pre_14 = pd.DataFrame(data)
        
        # Function to create HTML for the distribution bars with centering
        def create_distribution_bar(percentages, colors):
            bar_html = '<div style="width: 100%; height: 20px; display: flex; background: transparent;">'
            for percent, color in zip(percentages, colors):
                bar_html += f'<div style="flex: {percent}; background-color: {color}; height: 20px;"></div>'
            bar_html += '</div>'
            return bar_html
        
        # Apply the function to create the Distribution column with HTML bars
        df_pre_14['Distribution'] = df_pre_14['Distribution'].apply(lambda x: create_distribution_bar(x, colors))
        
        # Convert DataFrame to HTML
        def render_dataframe(df_pre_14):
            df_html = df_pre_14.to_html(escape=False, index=False)
            return df_html
        
        # Inject custom CSS to widen the Distribution column and center headers
        st.markdown("""
            <style>
            .dataframe th:nth-child(3), .dataframe td:nth-child(3) {
                min-width: 400px;
            }
            .dataframe th {
                text-align: center;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Display the table with distribution bars
        st.write(render_dataframe(df_pre_14), use_container_width=True, unsafe_allow_html=True)
    
    with col8_2:
        distribution_explain = "**Distribution:** A visual representation of how many times each choice was ranked. "
        score_explain = "**Score:** The first position gives the highest “weight” and the last position gives the lowest “weight”.\
            We calculate the total score of each choice based on these weighted values, and took the average to obtain the score."
        ranked_explain = "**Times Ranked:** This column counts the number of times this item is ranked as the top option."
        
        st.markdown(distribution_explain)
        st.markdown(score_explain)
        st.markdown(ranked_explain)

with tabs[2]:
    total_entries = len(df_post)
    
    st.header(":green[Response to Survey Questions]")
    st.subheader(":blue[Retrospective Survey]")
    col1_1, col1_2, col1_3 = st.columns((3,3,2))
    
    with col1_1:
        st.markdown("Based on what you now know, consider where you were :blue[**BEFORE**] using InPlace. Indicate your agreement to the following statements.")

        # List of specific columns to be plotted
        Question_1A = ['1.1.A', '1.1.B', '1.1.C', '1.1.D', '1.1.E', '1.1.F', '1.1.G', '1.1.H', '1.1.I', '1.1.J']
        
        # Create a mapping from original column names to new labels ("A", "B", "C", etc.)
        new_labels = [chr(65 + i) for i in range(len(Question_1A))]
        column_label_mapping = dict(zip(Question_1A, new_labels[::-1]))
        
        # Convert numerical responses to Likert scale labels
        likert_labels = {1: 'Strongly disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly agree'}
        
        # Initialize a DataFrame to store the frequency counts
        frequency_data_1 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        hover_data_1 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_1A:
            frequency_data_1['Question'].append(column_label_mapping[question])
            hover_data_1['Question'].append(column_label_mapping[question])
            counts = df_post[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_1[label].append(percentage)
                hover_data_1[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_post_1 = pd.DataFrame(frequency_data_1)
        hover_df_post_1 = pd.DataFrame(hover_data_1)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#f1f1f1', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_post_1['Question'],
                x=frequency_df_post_1[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_post_1[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_post_1['Question']) - 0.5,
            line=dict(
                color='Black',
                width=2,
                dash='dash',
            ),
        )
        
        fig_diverging.update_layout(
            barmode='relative',
            xaxis_title='Percentage (%)',
            yaxis_title='Questions',
            legend_title='Responses',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            width=600,  # Set the width of the figure
            height=500,   # Set the height of the figure
            legend=dict(traceorder='reversed')
            )
        
        st.plotly_chart(fig_diverging,use_container_width=True)
    
    with col1_2:
        st.markdown("Based on what you now know, consider where you were :red[**AFTER**] using InPlace. Indicate your agreement to the following statements.")

        # List of specific columns to be plotted
        Question_1B = ['1.2.A', '1.2.B', '1.2.C', '1.2.D', '1.2.E', '1.2.F', '1.2.G', '1.2.H', '1.2.I', '1.2.J']
        
        # Create a mapping from original column names to new labels ("A", "B", "C", etc.)
        new_labels = [chr(65 + i) for i in range(len(Question_1B))]
        column_label_mapping = dict(zip(Question_1B, new_labels[::-1]))
        
        # Convert numerical responses to Likert scale labels
        likert_labels = {1: 'Strongly disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly agree'}
        
        # Initialize a DataFrame to store the frequency counts
        frequency_data_1 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        hover_data_1 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_1B:
            frequency_data_1['Question'].append(column_label_mapping[question])
            hover_data_1['Question'].append(column_label_mapping[question])
            counts = df_post[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_1[label].append(percentage)
                hover_data_1[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_post_1 = pd.DataFrame(frequency_data_1)
        hover_df_post_1 = pd.DataFrame(hover_data_1)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#f1f1f1', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_post_1['Question'],
                x=frequency_df_post_1[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_post_1[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_post_1['Question']) - 0.5,
            line=dict(
                color='Black',
                width=2,
                dash='dash',
            ),
        )
        
        fig_diverging.update_layout(
            barmode='relative',
            xaxis_title='Percentage (%)',
            yaxis_title='Questions',
            legend_title='Responses',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            width=600,  # Set the width of the figure
            height=500,   # Set the height of the figure
            legend=dict(traceorder='reversed')
            )
        
        st.plotly_chart(fig_diverging,use_container_width=True)
    
    with col1_3:
        st.markdown("I had/have")
        # List of specific columns to be plotted
        Question_1_Description = [
            'knowledge of the skills I possessed.',
            'knowledge of the skills required for my desired industry.',
            'knowledge of the skills I do not have (skills gap) for my desired industry.',
            'a list of career options.',
            'a plan to prioritise the skills that I should develop.',
            'an action plan of the courses I should take.',
            'knowledge of resources that can help me research my career options.',
            'confidence in my ability to research career, employment, and available training.',
            'effective strategies to keep myself on track to achieve my educational and employment goals.',
            'confidence in my ability to manage future career changes.'
            ]
        
        mapping_table = pd.DataFrame({
        'Label': new_labels,
        'Survey Question': Question_1_Description
        })
        
        st.dataframe(mapping_table,height=(len(column_label_mapping)+1)*35+3,use_container_width=True,hide_index=True)

    col2_1, col2_2 = st.columns((6,2))
    with col2_1:
        Qn_Index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

        # Calculate difference scores
        for col in Qn_Index:
            df_post[f'{col}_diff'] = df_post[f'1.2.{col}'] - df_post[f'1.1.{col}']

        # Consolidate histograms
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
        axes = axes.flatten()
        for i, col in enumerate(Qn_Index):
            sns.histplot(df_post[f'{col}_diff'], kde=False, ax=axes[i])
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel('Difference Score')
            axes[i].set_ylabel('Frequency')
            axes[i].set_xlim(-4,4)
            axes[i].set_ylim(0,71)
        plt.tight_layout()
        st.pyplot(fig,use_container_width=True)

    with col2_2:
        # Paired t-test
        results = {}
        for col in Qn_Index:
            t_stat, p_val = stats.ttest_rel(df_post[f'1.2.{col}'], df_post[f'1.1.{col}'])
            results[col] = {'t_stat': t_stat, 'p_val': p_val}

        # Effect size (Cohen's d)
        def cohens_d(x, y):
            nx = len(x)
            ny = len(y)
            dof = nx + ny - 2
            return (x.mean() - y.mean()) / (((nx - 1) * x.var() + (ny - 1) * y.var()) / dof) ** 0.5

        effect_sizes = {}
        for col in Qn_Index:
            effect_sizes[col] = cohens_d(df_post[f'1.2.{col}'], df_post[f'1.1.{col}'])
        
        # Summary table
        summary_table = pd.DataFrame.from_dict(results, orient='index')
        summary_table['effect_size'] = pd.Series(effect_sizes)
        
        st.dataframe(summary_table,height=(len(column_label_mapping)+1)*35+3,use_container_width=True,hide_index=False)
    
    st.subheader(":blue[Construct Study of InPlace]")
    col3_1, col3_2 = st.columns((2,1))
    
    with col3_1:
        st.markdown("This question is about the **influence of InPlace** on course selection. Please rate your agreement with the following statements:")

        # List of specific columns to be plotted
        Question_4 = [
        '2.1 (Q4_A_1)', '2.1 (Q4_A_2)', '2.1 (Q4_A_3)', '2.1 (Q4_A_4)', '2.1 (Q4_A_5)', '2.1 (Q4_A_6)', '2.1 (Q4_A_7)', '2.1 (Q4_A_8)']
        
        # Create a mapping from original column names to new labels ("A", "B", "C", etc.)
        new_labels = [chr(65 + i) for i in range(len(Question_4))]
        column_label_mapping = dict(zip(Question_4, new_labels[::-1]))
        
        # Convert numerical responses to Likert scale labels
        likert_labels = {1: 'Strongly disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly agree'}
        
        # Initialize a DataFrame to store the frequency counts
        frequency_data_4 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        hover_data_4 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_4:
            frequency_data_4['Question'].append(column_label_mapping[question])
            hover_data_4['Question'].append(column_label_mapping[question])
            counts = df_post[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_4[label].append(percentage)
                hover_data_4[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_post_4 = pd.DataFrame(frequency_data_4)
        hover_df_post_4 = pd.DataFrame(hover_data_4)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#f1f1f1', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_post_4['Question'],
                x=frequency_df_post_4[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_post_4[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_post_4['Question']) - 0.5,
            line=dict(
                color='Black',
                width=2,
                dash='dash',
            ),
        )
        
        fig_diverging.update_layout(
            barmode='relative',
            xaxis_title='Percentage (%)',
            yaxis_title='Questions',
            legend_title='Responses',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            width=1100,  # Set the width of the figure
            height=400,   # Set the height of the figure
            legend=dict(traceorder='reversed')
            )
        
        st.plotly_chart(fig_diverging,use_container_width=True)
        
    with col3_2:
        # List of specific columns to be plotted
        Question_4_Description = [
            'InPlace provides useful information on available academic courses from NTU that can help narrow my skills gap.',
            'InPlace provides useful information on available cocurricular activities from NTU that can help narrow my skills gap.',
            'InPlace provides me with useful information on available soft skill courses, outside of NTU that can help narrow my skills gap.',
            'InPlace provides me with useful information on available functional skill courses, outside of NTU that can help narrow my skills gap.',
            'InPlace provides me with useful information on available domain skill courses, outside of NTU that can help narrow my skills gap.',
            'InPlace provides me with useful information on available requirement skill courses, outside of NTU that can help narrow my skills gap.',
            'InPlace facilitates my decision-making process on which course(s) I should take.',
            'InPlace facilitates my decision on the internship I should apply for.'
            ]
        
        mapping_table = pd.DataFrame({
        'Label': new_labels,
        'Survey Question': Question_4_Description
        })
        
        st.dataframe(mapping_table,height=(len(column_label_mapping)+1)*35+3,use_container_width=True,hide_index=True)
        
    col4_1, col4_2 = st.columns((2,1))
    
    with col4_1:
        st.markdown("This question is about the **the perceived usefulness of InPace** beyond your first. Please rate your agreement with the following statements:")

        # List of specific columns to be plotted
        Question_5 = ['2.2 (Q5_A_1)', '2.2 (Q5_A_2)', '2.2 (Q5_A_3)']
        
        # Create a mapping from original column names to new labels ("A", "B", "C", etc.)
        new_labels = [chr(65 + i) for i in range(len(Question_5))]
        column_label_mapping = dict(zip(Question_5, new_labels[::-1]))
        
        # Convert numerical responses to Likert scale labels
        likert_labels = {1: 'Strongly disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly agree'}
        
        # Initialize a DataFrame to store the frequency counts
        frequency_data_5 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        hover_data_5 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_5:
            frequency_data_5['Question'].append(column_label_mapping[question])
            hover_data_5['Question'].append(column_label_mapping[question])
            counts = df_post[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_5[label].append(percentage)
                hover_data_5[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_post_5 = pd.DataFrame(frequency_data_5)
        hover_df_post_5 = pd.DataFrame(hover_data_5)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#f1f1f1', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_post_5['Question'],
                x=frequency_df_post_5[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_post_4[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_post_5['Question']) - 0.5,
            line=dict(
                color='Black',
                width=2,
                dash='dash',
            ),
        )
        
        fig_diverging.update_layout(
            barmode='relative',
            xaxis_title='Percentage (%)',
            yaxis_title='Questions',
            legend_title='Responses',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            width=1100,  # Set the width of the figure
            height=400,   # Set the height of the figure
            legend=dict(traceorder='reversed')
            )
        
        st.plotly_chart(fig_diverging,use_container_width=True)
        
    with col4_2:
        st.markdown("If InPlace is made accessible to me beyond graduation,")
        # List of specific columns to be plotted
        Question_5_Description = [
            'I will use it in the future to find out how to narrow my skills gap within my desired industry.',
            'I will use it in the future to find out how to narrow my skills gap to facilitate a job change to another industry.',
            'I will recommend other NTU students and alumni to use it.'
            ]
        
        mapping_table = pd.DataFrame({
        'Label': new_labels,
        'Survey Question': Question_5_Description
        })
        
        st.dataframe(mapping_table,height=(len(column_label_mapping)+1)*35+3,use_container_width=True,hide_index=True)
    
    col5_1, col5_2 = st.columns((2,1))
    
    with col5_1:
        st.markdown("This question is about the **the perceived usefulness of InPace** beyond your first. Please rate your agreement with the following statements:")

        # List of specific columns to be plotted
        Question_6 = ['3.1 (Q6_A_1)', '3.1 (Q6_A_2)', '3.1 (Q6_A_3)', '3.1 (Q6_A_4)', '3.1 (Q6_A_5)', '3.1 (Q6_A_6)', '3.1 (Q6_A_7)']
        
        # Create a mapping from original column names to new labels ("A", "B", "C", etc.)
        new_labels = [chr(65 + i) for i in range(len(Question_6))]
        column_label_mapping = dict(zip(Question_6, new_labels[::-1]))
        
        # Convert numerical responses to Likert scale labels
        likert_labels = {1: 'Strongly disagree', 2: 'Disagree', 3: 'Neutral', 4: 'Agree', 5: 'Strongly agree'}
        
        # Initialize a DataFrame to store the frequency counts
        frequency_data_6 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        hover_data_6 = {'Question': [], 'Strongly disagree': [], 'Disagree': [], 'Neutral': [], 'Agree': [], 'Strongly agree': []}
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_6:
            frequency_data_6['Question'].append(column_label_mapping[question])
            hover_data_6['Question'].append(column_label_mapping[question])
            counts = df_post[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_6[label].append(percentage)
                hover_data_6[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_post_6 = pd.DataFrame(frequency_data_6)
        hover_df_post_6 = pd.DataFrame(hover_data_6)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#f1f1f1', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_post_6['Question'],
                x=frequency_df_post_6[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_post_4[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_post_6['Question']) - 0.5,
            line=dict(
                color='Black',
                width=2,
                dash='dash',
            ),
        )
        
        fig_diverging.update_layout(
            barmode='relative',
            xaxis_title='Percentage (%)',
            yaxis_title='Questions',
            legend_title='Responses',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            width=1100,  # Set the width of the figure
            height=400,   # Set the height of the figure
            legend=dict(traceorder='reversed')
            )
        
        st.plotly_chart(fig_diverging,use_container_width=True)
        
    with col5_2:
        # List of specific columns to be plotted
        Question_6_Description = [
            'InPlace is user-friendly and easy to use.',
            'I can intuitively navigate through InPlace.',
            'I did not face any technical difficulties when using InPlace.',
            'InPlace loads and operates at a speed I found satisfactory.',
            'I am overall satisfied with the usability of InPlace.',
            'I trust InPlace to correctly analyse my current skills.',
            'I trust InPlace to correctly evaluate the skills required for the current job market.'
            ]
        
        mapping_table = pd.DataFrame({
        'Label': new_labels,
        'Survey Question': Question_6_Description
        })
        
        st.dataframe(mapping_table,height=(len(column_label_mapping)+1)*35+3,use_container_width=True,hide_index=True)
with tabs[3]:
    st.subheader("Question Cluster Analysis of Factors Influencing Students' Elective Choice by Discipline")
    st.markdown("This section show the results when we performs hierarchical clustering on survey data from STEM and SHAPE disciplines, \
    calculates mean scores for each cluster, and performs a Mann-Whitney U test on paired clusters.")
    
    df_pre = pd.read_excel("PRE_RAW_COMPLETED_240711.xlsx")
    
    col1_1, col1_2 = st.columns((1,1))
    with col1_1:
        columns = ['2.1 (Q4_A_14)', '2.1 (Q4_A_13)', '2.1 (Q4_A_12)', '2.1 (Q4_A_11)',
            '2.1 (Q4_A_10)', '2.1 (Q4_A_9)', '2.1 (Q4_A_8)', '2.1 (Q4_A_7)',
            '2.1 (Q4_A_6)', '2.1 (Q4_A_5)', '2.1 (Q4_A_4)', '2.1 (Q4_A_3)',
            '2.1 (Q4_A_2)', '2.1 (Q4_A_1)']
        # Separate STEM and SHAPE Disciplines
        stem_data = df_pre[df_pre['Discipline'] == 'STEM'][columns]
        shape_data = df_pre[df_pre['Discipline'] == 'SHAPE'][columns]
        
        # Relabel columns to A-N
        new_column_labels = [chr(i) for i in range(ord('A'), ord('A') + len(columns))]
        stem_data.columns = new_column_labels
        shape_data.columns = new_column_labels
        
        # Transpose the data to cluster questions (columns)
        stem_data_transposed = stem_data.T
        shape_data_transposed = shape_data.T
        
        # Hierarchical clustering for STEM Discipline questions
        stem_linkage = linkage(stem_data_transposed, method='ward', metric='euclidean')
        
        # Hierarchical clustering for SHAPE Discipline questions
        shape_linkage = linkage(shape_data_transposed, method='ward', metric='euclidean')
        
        # Define number of clusters, k+1
        k = st.slider('Select number of clusters', 1, 14, 4)
        k = k-1
        
        # Plotting dendrograms
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Dendrogram for STEM group questions
        dendrogram(
            stem_linkage,
            ax=axes[0],
            labels=stem_data_transposed.index,
            color_threshold=stem_linkage[-k, 2]  # Color the top k+1 clusters
        )
        axes[0].set_title('STEM Discipline')
        axes[0].tick_params(axis='y', which='both', left=False, labelleft=False)
        
        # Dendrogram for SHAPE group questions
        dendrogram(
            shape_linkage,
            ax=axes[1],
            labels=shape_data_transposed.index,
            color_threshold=shape_linkage[-k, 2]  # Color the top k+1 clusters
        )
        axes[1].set_title('SHAPE Discipline')
        axes[1].tick_params(axis='y', which='both', left=False, labelleft=False)
    
        st.pyplot(fig,use_container_width=True)

    with col1_2:
        st.subheader("Mann-Whitney U Test Results for Nearest Cluster Pairings:")
        col1_2_1, col1_2_2, col1_2_3 = st.columns((1,1,1))

        with col1_2_1:
            k = k+1
            # Get cluster assignments for STEM and SHAPE groups
            stem_clusters = fcluster(stem_linkage, k, criterion='maxclust')
            shape_clusters = fcluster(shape_linkage, k, criterion='maxclust')
        
            # Map cluster labels to questions for STEM
            stem_cluster_members = {i: [] for i in range(1, k + 1)}
            for question, cluster in zip(stem_data_transposed.index, stem_clusters):
                stem_cluster_members[cluster].append(question)
        
            st.markdown("**STEM Cluster Members:**")
            for cluster, members in stem_cluster_members.items():
                st.write(f"Cluster {cluster}: {', '.join(members)}")
        
            # Map cluster labels to questions for SHAPE
            shape_cluster_members = {i: [] for i in range(1, k + 1)}
            for question, cluster in zip(shape_data_transposed.index, shape_clusters):
                shape_cluster_members[cluster].append(question)
        
            st.markdown("**SHAPE Cluster Members:**")
            for cluster, members in shape_cluster_members.items():
                st.write(f"Cluster {cluster}: {', '.join(members)}")
    
        with col1_2_2:
            # Calculate mean scores for each cluster
            stem_cluster_means = {}
            shape_cluster_means = {}
        
            for cluster in range(1, k + 1):
                stem_cluster_data = stem_data[stem_cluster_members[cluster]]
                shape_cluster_data = shape_data[shape_cluster_members[cluster]]
        
                stem_cluster_mean = stem_cluster_data.mean().mean()
                shape_cluster_mean = shape_cluster_data.mean().mean()
        
                stem_cluster_means[cluster] = stem_cluster_mean
                shape_cluster_means[cluster] = shape_cluster_mean
        
            # Print mean scores for each cluster
            st.markdown("**Mean Scores for STEM Clusters:**")
            for cluster, mean_score in stem_cluster_means.items():
                st.write(f"Cluster {cluster}: {mean_score:.4f}")
        
            st.markdown("**Mean Scores for SHAPE Clusters:**")
            for cluster, mean_score in shape_cluster_means.items():
                st.write(f"Cluster {cluster}: {mean_score:.4f}")
        
            # Calculate the distance between the mean scores of each cluster using Manhattan distance
            stem_means = np.array(list(stem_cluster_means.values())).reshape(-1, 1)
            shape_means = np.array(list(shape_cluster_means.values())).reshape(-1, 1)
        
            distances = cdist(stem_means, shape_means, metric='cityblock')  # Using Manhattan distance (cityblock)
        
            # Find the nearest cluster in SHAPE for each cluster in STEM
            nearest_clusters = np.argmin(distances, axis=1)
    
        with col1_2_3:    
            # Mapping the clusters with minimal distance
            cluster_pairings = {}
            for i, shape_cluster in enumerate(nearest_clusters):
                stem_cluster = i + 1
                cluster_pairings[stem_cluster] = shape_cluster + 1
        
            # Perform Mann-Whitney U test on all records in the cluster pairings
            mannwhitney_results = {}
            for stem_cluster, shape_cluster in cluster_pairings.items():
                stem_cluster_data = stem_data[stem_cluster_members[stem_cluster]]
                shape_cluster_data = shape_data[shape_cluster_members[shape_cluster]]
        
                # Flatten the data for Mann-Whitney U test
                stem_values = stem_cluster_data.values.flatten()
                shape_values = shape_cluster_data.values.flatten()
        
                # Perform the Mann-Whitney U test
                stat, p_value = mannwhitneyu(stem_values, shape_values, alternative='two-sided')
                
                # Calculate the effect size
                n1 = len(stem_values)
                n2 = len(shape_values)
                N = n1 + n2
                z = norm.ppf(p_value / 2) if p_value != 0 else 0  # Convert p-value to Z-score
                r = z / np.sqrt(N)  # Effect size
                
                mannwhitney_results[(stem_cluster, shape_cluster)] = (stat, p_value, r)

            for clusters, result in mannwhitney_results.items():
                stem_cluster, shape_cluster = clusters
                stat, p_value, effect_size = result

                if p_value <=0.05:
                    st.markdown(f"**:red[STEM Cluster {stem_cluster} vs SHAPE Cluster {shape_cluster}:]**")
                    st.write(f"  - U statistic: {stat:.4f}")
                    st.write(f"  - P-value: {p_value:.4f}")
                    st.write(f"  - Effect size (r): {effect_size:.4f}")
                #else:
                #    st.markdown(f"**STEM Cluster {stem_cluster} vs SHAPE Cluster {shape_cluster}:**")
                #    st.write(f"  - U statistic: {stat:.4f}")
                #    st.write(f"  - P-value: {p_value:.4f}")
                #    st.write(f"  - Effect size (r): {effect_size:.4f}")
    
    
