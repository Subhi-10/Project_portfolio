# Install the Cohere package
!pip install cohere
import cohere

# Initialize the Cohere client with your API key
api_key = 'Bm9bXtgKphHWiy1FGkJaErDbSRhyLCmik5V5esLU'  # Your API key
co = cohere.Client(api_key)

# Function to generate a crime and investigation-related question
def generate_question():
    prompt = "Generate a question related to crime and investigation with 2 options formatted as: Question? (Option 1, Option 2)\n"
   
    # Generate a question using Cohere API
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        num_generations=1
    )
   
    # Return the generated question
    return response.generations[0].text.strip()

# Initialize counters for assistant choices
assistant_1_votes = 0
assistant_2_votes = 0

# Generate questions one by one
try:
    for i in range(7):  # Generate 7 questions
        question = generate_question()
        print(f"Question {i + 1}: {question}")
       
        # Extract options from the generated question
        options = question.split('(')[-1].strip(')')  # Get options part
        option_list = options.split(', ')
       
        # Display the options to the user
        print("Choose an option:")
        print(f"1. {option_list[0]} ")
        print(f"2. {option_list[1]} ")
       
        # Get user's answer
        answer = input("Enter 1 or 2: ").strip()
       
        # Update votes based on user's choice
        if answer == '1':
            assistant_1_votes += 1
        elif answer == '2':
            assistant_2_votes += 1
        else:
            print("Invalid input. Please enter 1 or 2.")

    # Calculate percentages
    total_questions = 7
    assistant_1_percentage = (assistant_1_votes / total_questions) * 100
    assistant_2_percentage = (assistant_2_votes / total_questions) * 100

    # Determine the selected assistant based on percentages
    if assistant_1_percentage > assistant_2_percentage:
        selected_assistant = "Assistant 1"
    elif assistant_2_percentage > assistant_1_percentage:
        selected_assistant = "Assistant 2"
    else:
        selected_assistant = "It's a tie!"

    # Display results
    print(f"\nAssistant 1 Score: {assistant_1_percentage:.2f}%")
    print(f"Assistant 2 Score: {assistant_2_percentage:.2f}%")
    print(f"\nSelected Assistant: {selected_assistant}")

except Exception as e:
    print("Error:", str(e))
