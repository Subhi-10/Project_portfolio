import pygame
import random
import re
import cohere
import time
import sys

# Initialize Cohere client (replace with your API key)
co = cohere.Client("Bm9bXtgKphHWiy1FGkJaErDbSRhyLCmik5V5esLU")

# Initialize Pygame
pygame.init()

# Display Settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mystery Word Guessing Game")

# Colors
BACKGROUND_COLOR = (40, 40, 60)
TEXT_COLOR = (255, 255, 255)
HINT_COLOR = (255, 215, 0)
INPUT_BOX_COLOR = (80, 120, 180)
BUTTON_COLOR = (60, 180, 75)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Fonts
FONT_LARGE = pygame.font.Font(None, 60)
FONT_MEDIUM = pygame.font.Font(None, 40)
FONT_SMALL = pygame.font.Font(None, 30)

# FPS
clock = pygame.time.Clock()

def generate_word():
    """Generate a word using Cohere."""
    themes = ["mystery", "crime", "investigation", "crime weapons", "crime motives"]
    selected_theme = random.choice(themes)
    prompt = f"Generate a simple word (max 12 letters) related to {selected_theme}."
    
    try:
        response = co.generate(
            model="command-xlarge-nightly",
            prompt=prompt,
            max_tokens=10,
            temperature=0.9,
        )
        word = response.generations[0].text.strip().lower()
        return re.sub(r'[^a-z]', '', word) if 3 <= len(word) <= 12 else "clue"
    except Exception as e:
        print(f"Error generating word: {e}")
        return "clue"

def generate_predefined_hints(word):
    """Predefined hints based on the word."""
    return [
        f"The word starts with '{word[0]}'.",
        f"The word ends with '{word[-1]}'.",
        f"The word has {len(word)} letters.",
        f"It contains {sum(1 for c in word if c in 'aeiou')} vowels."
    ]

def generate_cohere_hint(word):
    """Generate a hint using Cohere for the given word."""
    prompt = f"Provide a simple hint for the word '{word}' suitable for a child. The hint should be short and easy to understand, also not include the word itself."
    try:
        response = co.generate(
            model="command-xlarge-nightly",
            prompt=prompt,
            max_tokens=20,
            temperature=0.7,
        )
        return response.generations[0].text.strip()
    except Exception as e:
        print(f"Error generating Cohere hint: {e}")
        return "No hint available."

def provide_feedback(word, guess):
    """Provide feedback in terms of gold and silver stars."""
    gold = sum(1 for i in range(min(len(guess), len(word))) if guess[i] == word[i])
    silver = sum(1 for c in guess if c in word) - gold
    return f"Gold: {'⭐' * gold}, Silver: {'⚪' * silver}"

def draw_text(text, font, color, x, y, center=True):
    """Draw text on the screen."""
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(x, y) if center else (x, y))
    screen.blit(surface, rect)

def draw_button(text, x, y, width, height):
    """Draw a button on the screen."""
    pygame.draw.rect(screen, BUTTON_COLOR, (x, y, width, height))
    draw_text(text, FONT_SMALL, BUTTON_TEXT_COLOR, x + width // 2, y + height // 2)

def reveal_random_letter(word, revealed):
    """Reveal a random letter (not the first or last) and its position."""
    indices = [i for i in range(1, len(word) - 1) if i not in revealed]
    if not indices:
        return "No more letters to reveal."
    random_index = random.choice(indices)
    revealed.add(random_index)
    return f"Letter at position {random_index + 1}: '{word[random_index]}'"

def play_game():
    """Main game loop."""
    word = generate_word()
    predefined_hints = generate_predefined_hints(word)
    cohere_hints = [generate_cohere_hint(word) for _ in range(2)]
    all_hints = predefined_hints + cohere_hints
    hint_index = 0
    wildcard_uses = 2
    guesses = 0
    max_guesses = 10
    input_text = ""
    message = ""
    game_over = False
    revealed_letters = set()

    while True:
        screen.fill(BACKGROUND_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if input_text.lower() == word:
                        message = "Congratulations! You guessed it!"
                        game_over = True
                    else:
                        feedback = provide_feedback(word, input_text.lower())
                        guesses += 1
                        input_text = ""
                        if guesses <= max_guesses:
                            hint = all_hints[hint_index % len(all_hints)]
                            hint_index += 1
                            message = f"Feedback: {feedback} | Hint: {hint}"
                        else:
                            message = f"Feedback: {feedback} | No more hints."
                        if guesses >= max_guesses:
                            message = f"Game Over! The word was '{word}'."
                            game_over = True
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if wildcard_button.collidepoint(event.pos) and wildcard_uses > 0:
                    wildcard_hint = reveal_random_letter(word, revealed_letters)
                    message = f"Wildcard Hint: {wildcard_hint}"
                    wildcard_uses -= 1

        draw_text("Mystery Word Guessing Game", FONT_LARGE, TEXT_COLOR, WIDTH // 2, 50)
        draw_text(f"Guesses: {guesses}/{max_guesses}", FONT_SMALL, TEXT_COLOR, WIDTH - 150, 20, center=False)

        input_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 - 30, 300, 60)
        pygame.draw.rect(screen, INPUT_BOX_COLOR, input_rect, 2)
        draw_text(input_text, FONT_MEDIUM, TEXT_COLOR, input_rect.centerx, input_rect.centery)

        if message:
            feedback_line, *hint_lines = message.split(" | Hint: ")
            draw_text(feedback_line, FONT_SMALL, HINT_COLOR, WIDTH // 2, HEIGHT - 150)
            for i, line in enumerate(hint_lines):
                draw_text(line, FONT_SMALL, HINT_COLOR, WIDTH // 2, HEIGHT - 120 + (i * 30))

        wildcard_button = pygame.Rect(WIDTH // 2 - 75, HEIGHT - 80, 150, 50)
        draw_button(f"Wildcard ({wildcard_uses})", wildcard_button.x, wildcard_button.y, wildcard_button.width, wildcard_button.height)

        pygame.display.flip()
        clock.tick(30)

play_game()
pygame.quit()
