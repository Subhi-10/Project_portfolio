import pygame
import sys
import random
import re
import cohere

# Initialize Pygame and Cohere API
pygame.init()
co = cohere.Client("Bm9bXtgKphHWiy1FGkJaErDbSRhyLCmik5V5esLU")  # Replace with your API key

# Set up display constants
WIDTH, HEIGHT = 800, 600
FONT_LARGE = pygame.font.Font(None, 48)
FONT_MEDIUM = pygame.font.Font(None, 36)
FONT_SMALL = pygame.font.Font(None, 24)

# Colors
BACKGROUND = (30, 30, 50)
TEXT_COLOR = (255, 255, 255)
HINT_COLOR = (200, 200, 100)
INPUT_BOX_COLOR = (100, 100, 200)
INPUT_TEXT_COLOR = (255, 255, 255)

# Function to filter out non-alphabet characters from a word
def filter_word(word):
    return ''.join(re.findall(r'[a-zA-Z]', word))

# Function to display the word with dashes only for letters (no extra dashes)
def display_word_with_dashes(word, guessed_letters):
    return ''.join(
        char if not char.isalpha() or char in guessed_letters else '_' 
        for char in word
    )

# Function to generate hints based on word properties
def generate_hint(word, hint_type):
    filtered_word = filter_word(word).lower()
    if hint_type == "first_letter":
        return f"It starts with '{filtered_word[0]}'"
    elif hint_type == "last_letter":
        return f"It ends with '{filtered_word[-1]}'"
    elif hint_type == "num_vowels":
        vowel_count = sum(1 for char in filtered_word if char in 'aeiou')
        return f"It has {vowel_count} vowels"
    elif hint_type == "word_length":
        return f"It's {len(filtered_word)} letters long"
    elif hint_type == "middle_letter":
        mid = len(filtered_word) // 2
        return f"The middle letter is '{filtered_word[mid]}'"

# Generate a random crime-related word using Cohere
def generate_crime_related_word():
    variations = [
        "Give a new word each time related to crime, mystery, or investigation.",
        "Suggest a simple crime-related word for this game.",
        "Provide a unique word linked to evidence, crime, or detective work."
    ]
    prompt = random.choice(variations)
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=10,
        temperature=0.9  # Increase randomness
    )
    return response.generations[0].text.strip()

# Generate hints using Cohere
def generate_cohere_hints(word, num_hints=4):
    prompt = f"Give {num_hints} very simple hints about '{word}' for a 5-year-old playing a guessing game. The hints should be short, easy to understand, and not use the word itself. Separate the hints with newlines."
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    return response.generations[0].text.strip().split('\n')

def draw_text(surface, text, font, color, x, y, center=True):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center:
        text_rect.center = (x, y)
    else:
        text_rect.topleft = (x, y)
    surface.blit(text_surface, text_rect)

def wrap_text(text, font, max_width):
    words = text.split(' ')
    wrapped_lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if font.size(' '.join(current_line))[0] > max_width:
            if len(current_line) > 1:
                current_line.pop()
                wrapped_lines.append(' '.join(current_line))
                current_line = [word]
            else:
                wrapped_lines.append(word)
                current_line = []
    if current_line:
        wrapped_lines.append(' '.join(current_line))
    return wrapped_lines

def play_game():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Word Guessing Game")

    word = generate_crime_related_word().lower()
    guessed_letters = set()
    max_guesses = 10
    total_guesses = 0
    
    predefined_hints = [
        generate_hint(word, "first_letter"),
        generate_hint(word, "last_letter"),
        generate_hint(word, "num_vowels"),
        generate_hint(word, "word_length"),
        generate_hint(word, "middle_letter")
    ]
    
    cohere_hints = generate_cohere_hints(word)
    all_hints = predefined_hints + cohere_hints
    random.shuffle(all_hints)

    current_hint_index = 0

    input_box = pygame.Rect(300, 500, 200, 32)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    
    color = color_inactive
    active = False
    text = ''
    
    message = ""
    
    game_over = False

    clock = pygame.time.Clock()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive
            
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        guess = text.lower()
                        text = ''
                        if guess == filter_word(word) or set(filter_word(word)) <= guessed_letters:
                            message = f"Congratulations! You guessed the word: {word}!"
                            game_over = True
                        elif len(guess) == 1 and guess.isalpha():
                            if total_guesses < max_guesses: 
                                total_guesses += 1
                                if guess in word:
                                    guessed_letters.add(guess)
                                    message = f"Good guess! '{guess}' is in the word."
                                else:
                                    message = f"Wrong guess! You have {max_guesses - total_guesses} guesses left."

                                current_hint_index += 1

                                # Check if all letters have been guessed correctly 
                                if set(filter_word(word)) <= guessed_letters:
                                    message += f" Congratulations! You guessed the word: {word}!"
                                    game_over = True

                            if total_guesses >= max_guesses: 
                                message += f" Game Over! The word was: {word}."
                                game_over = True
                        else:
                            message += " Invalid input! Try again."
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode

        screen.fill(BACKGROUND)

        # Draw word
        display_word = display_word_with_dashes(word, guessed_letters)
        draw_text(screen, display_word, FONT_LARGE, TEXT_COLOR, WIDTH // 2, 100)

        # Draw wrapped hint
        if current_hint_index < len(all_hints):
            hint_lines = wrap_text(f"Hint: {all_hints[current_hint_index]}", FONT_MEDIUM, WIDTH - 100)
            for i, line in enumerate(hint_lines):
                draw_text(screen, line, FONT_MEDIUM, HINT_COLOR, WIDTH // 2, 200 + i * 30)
        
        # Draw guesses left
        draw_text(screen, f"Guesses left: {max_guesses - total_guesses}", FONT_MEDIUM, TEXT_COLOR, WIDTH // 2, 300)

        # Draw message
        message_lines = wrap_text(message, FONT_MEDIUM, WIDTH - 100)
        for i, line in enumerate(message_lines):
            draw_text(screen, line, FONT_MEDIUM, TEXT_COLOR, WIDTH // 2, 400 + i * 30)

        # Draw input box
        txt_surface = FONT_MEDIUM.render(text, True, INPUT_TEXT_COLOR)
        width = max(200, txt_surface.get_width() + 10)
        
        input_box.w = width
        
        pygame.draw.rect(screen, color_inactive if not active else color_active , input_box)
        
        screen.blit(txt_surface,(input_box.x +5,input_box.y +5))

        
        pygame.display.flip()
        
        clock.tick(30)

# Start the game
play_game()
