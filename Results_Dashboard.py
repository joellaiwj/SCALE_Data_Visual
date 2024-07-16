import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Function to extract abbreviation from text in brackets
def extract_abbreviation(text):
    start = text.find('(')
    end = text.find(')')
    if start != -1 and end != -1 and start < end:
        return text[start+1:end]
    return text

st.set_page_config(page_title="SCALE Analysis",page_icon=":bar_chart:",layout="wide")

st.title(":bar_chart: SCALE Analysis")

file = "PRE_RAW_COMPLETED_240711.xlsx"
df = pd.read_excel(file)

tab3, tab1, tab2 = st.tabs([":male-technologist: Data", ":large_green_circle: Pre-Intervetion", ":large_orange_circle: Post-Intervention"])

# Extract abbreviations for "College" and "School" columns
df['College'] = df['College'].apply(extract_abbreviation)
df['School'] = df['School'].apply(extract_abbreviation)


with tab3:
    st.header(":green[Pre-Intervention Raw Data]")
    st.dataframe(df)

with tab1:
    st.header(":green[Participant Demographic]")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ##### COLLEGE #####
        st.subheader(":blue[College:]")
        counts_college = df['College'].value_counts().reset_index()
        counts_college.columns = ['College', 'Count']
        fig_college = px.pie(counts_college, values='Count', names='College')
        fig_college.update_layout(legend=dict(x=0.1,y=1,traceorder='normal'))
        st.plotly_chart(fig_college)
    
    with col2:
        ##### SCHOOL #####
        st.subheader(":blue[School:]")
        counts_school = df['School'].value_counts().reset_index()
        counts_school.columns = ['School', 'Count']
        fig_school = px.pie(counts_school, values='Count', names='School')
        fig_school.update_layout(legend=dict(x=0.0,y=1,traceorder='normal'))
        st.plotly_chart(fig_school)
        
    with col3:
        ##### YEAR OF STUDY #####
        st.subheader(":blue[Year of Study:]")
        counts_study = df['Year'].value_counts().reset_index()
        counts_study.columns = ['Year', 'Count']
        fig_study = px.pie(counts_study, values='Count', names='Year')
        fig_study.update_layout(legend=dict(x=0.75,y=1,traceorder='normal'))
        st.plotly_chart(fig_study)
    
    st.header(":green[Response to Survey Questions]")
    
    col4, col5 = st.columns((2,1))
    
    with col4:
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
        
        # Total number of entries
        total_entries = 154  # Replace with the actual number of entries in your data
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_4:
            frequency_data_4['Question'].append(column_label_mapping[question])
            hover_data_4['Question'].append(column_label_mapping[question])
            counts = df[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_4[label].append(percentage)
                hover_data_4[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_4 = pd.DataFrame(frequency_data_4)
        hover_df_4 = pd.DataFrame(hover_data_4)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_4['Question'],
                x=frequency_df_4[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_4[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_4['Question']) - 0.5,
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
        
        st.plotly_chart(fig_diverging)
    
    with col5:
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
        
        st.dataframe(mapping_table,height=3*len(df)+65,width=3*len(df),hide_index=True)
    
    col6, col7 = st.columns((20,1))
    
    with col6:
        st.subheader(":blue[Industry and Skill Requirements:]")
        st.markdown("Which industry is the FIRST (1st) choice for your first job after graduation?")
        
        ##### INDUSTRY #####
        # Columns to plot
        columns = ['College', 'School', 'Programme', 'Industry']
        #columns = ['School', 'Programme', 'Industry']
        
        # Count occurrences of each combination of "College", "School", "Programme", and "Industry"
        df_counts = df.groupby(columns).size().reset_index(name='count')
        
        # Prepare data for the Sankey diagram
        all_nodes = list(pd.concat([df[col] for col in columns]).unique())
        node_indices = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Create the Sankey diagram links
        links = []
        for _, row in df_counts.iterrows():
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
        
        st.plotly_chart(fig)
    
    
    # Get unique values for the filter column
    filter_values = df['Industry'].unique()
    
    # Multiselect for selecting filter values
    selected_values = st.multiselect("Filter by Industry", filter_values, default=filter_values)
    
    # Define stopwords
    stopwords = set(STOPWORDS)
    stopwords.add("skill")
    stopwords.add("skills")
    
    col8, col9 = st.columns(2)
    
    with col8:
        st.markdown("For my desired first choice, the skills that are important (in my opinion) are:")
        if selected_values:
            # Filter the DataFrame based on the selected values
            filtered_df = df[df['Industry'].isin(selected_values)]
    
            if not filtered_df.empty:
                # Combine all text from the three specific columns into one single string
                columns = ['3.2 (Q10_1)', '3.2 (Q10_2)', '3.2 (Q10_3)']
                combined_text = " ".join(filtered_df[columns[0]].astype(str)).lower() + " " + " ".join(filtered_df[columns[1]].astype(str)).lower() + " " + " ".join(filtered_df[columns[2]].astype(str)).lower()
    
                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(combined_text)
    
                # Display the generated word cloud image
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
        
    with col9:
        st.markdown("The important skills that I currently lack are:")
        if selected_values:
            # Filter the DataFrame based on the selected values
            filtered_df = df[df['Industry'].isin(selected_values)]
    
            if not filtered_df.empty:
                # Combine all text from the three specific columns into one single string
                columns = ['3.4 (Q12_1)', '3.4 (Q12_2)', '3.4 (Q12_3)']
                combined_text = " ".join(filtered_df[columns[0]].astype(str)).lower() + " " + " ".join(filtered_df[columns[1]].astype(str)).lower() + " " + " ".join(filtered_df[columns[2]].astype(str)).lower()
    
                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(combined_text)
    
                # Display the generated word cloud image
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
    
    col10, col11 = st.columns((2,1))
    
    with col10:
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
        
        # Total number of entries
        total_entries = 154  # Replace with the actual number of entries in your data
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_11:
            frequency_data_11['Question'].append(column_label_mapping[question])
            hover_data_11['Question'].append(column_label_mapping[question])
            counts = df[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_11[label].append(percentage)
                hover_data_11[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_11 = pd.DataFrame(frequency_data_11)
        hover_df_11 = pd.DataFrame(hover_data_11)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#f1f1f1', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_11['Question'],
                x=frequency_df_11[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_11[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_11['Question']) - 0.5,
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
        
        st.plotly_chart(fig_diverging)
        
    with col11:
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
        
        st.dataframe(mapping_table,height=2*len(df)+80,width=3*len(df),hide_index=True)
        
    col12, col13 = st.columns((2,1))
    
    with col12:
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
        
        # Total number of entries
        total_entries = 154  # Replace with the actual number of entries in your data
        
        # Count the frequency of each Likert response for each question and convert to percentage
        for question in Question_13:
            frequency_data_13['Question'].append(column_label_mapping[question])
            hover_data_13['Question'].append(column_label_mapping[question])
            counts = df[question].value_counts().sort_index()
            for key, label in likert_labels.items():
                count = counts.get(key, 0)
                percentage = (count / total_entries) * 100
                frequency_data_13[label].append(percentage)
                hover_data_13[label].append(count)
        
        # Create the frequency DataFrame
        frequency_df_13 = pd.DataFrame(frequency_data_13)
        hover_df_13 = pd.DataFrame(hover_data_13)
        
        # Create the diverging bar chart
        fig_diverging = go.Figure()
        
        colors = ['#de425b', '#f3babc', '#f1f1f1', '#aecdc2', '#488f31']
        
        for label, color in zip(likert_labels.values(), colors):
            fig_diverging.add_trace(go.Bar(
                y=frequency_df_13['Question'],
                x=frequency_df_13[label],
                name=label,
                orientation='h',
                marker=dict(color=color),
                text=hover_df_13[label],
                hovertemplate='%{text} responses'
            ))
        
        
        # Add a vertical dashed line at 50%
        fig_diverging.add_shape(
            type='line',
            x0=50,
            y0=-0.5,
            x1=50,
            y1=len(frequency_df_13['Question']) - 0.5,
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
        
        st.plotly_chart(fig_diverging)
        
    with col13:
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
        
        st.dataframe(mapping_table,height=len(df)-10,width=3*len(df),hide_index=True)

    col14, col15 = st.columns((2,1))
    
    with col14:
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
        colors = ['#000000', '#de425b', '#f3babc', '#aecdc2', '#488f31', '#000000']
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Function to create HTML for the distribution bars with centering
        def create_distribution_bar(percentages, colors):
            bar_html = '<div style="width: 100%; height: 20px; display: flex; background: transparent;">'
            for percent, color in zip(percentages, colors):
                bar_html += f'<div style="flex: {percent}; background-color: {color}; height: 20px;"></div>'
            bar_html += '</div>'
            return bar_html
        
        # Apply the function to create the Distribution column with HTML bars
        df['Distribution'] = df['Distribution'].apply(lambda x: create_distribution_bar(x, colors))
        
        # Convert DataFrame to HTML
        def render_dataframe(df):
            df_html = df.to_html(escape=False, index=False)
            return df_html
        
        # Inject custom CSS to widen the Distribution column and center headers
        st.markdown("""
            <style>
            .dataframe th:nth-child(3), .dataframe td:nth-child(3) {
                min-width: 700px;
            }
            .dataframe th {
                text-align: center;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Display the table with distribution bars
        st.write(render_dataframe(df), unsafe_allow_html=True)
    
    with col15:
        distribution_explain = "**Distribution:** A visual representation of how many times each choice was ranked. "
        score_explain = "**Score:** The first position gives the highest “weight” and the last position gives the lowest “weight”.\
            We calculate the total score of each choice based on these weighted values, and took the average to obtain the score."
        ranked_explain = "**Times Ranked:** This column counts the number of times this item is ranked as the top option."
        
        st.markdown(distribution_explain)
        st.markdown(score_explain)
        st.markdown(ranked_explain)

with tab2:
    st.subheader(":red[In Progress...]")
