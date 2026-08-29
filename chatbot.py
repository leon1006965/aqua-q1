import random
import re

class SimpleAI:
    def __init__(self):
        self.name = "Neural"
        self.markov = {}
        self.knowledge = self._load_knowledge()
        self._train_markov()

    def _load_knowledge(self):
        return {
            "roblox": "Roblox is a online platform where people create and play games made by other users. It has millions of games across every genre - from obbys (obstacle courses) to simulators to role-playing games. Users can also create their own games using Roblox Studio and earn money through their creations. It's especially popular with younger players but has a huge community of all ages.",
            "hello": f"Hey there! I'm {self.name}, a simple AI. I can chat and answer questions.",
            "hi": f"Hi! I'm {self.name}. What's up?",
            "hey": "Hey! What can I help with?",
            "who are you": "I'm a simple AI chatbot. I generate responses based on patterns I've learned. Not the smartest, but I try!",
            "what are you": "I'm a locally-running AI assistant. No cloud, no API - just pure Python and some clever text generation.",
            "how do you work": "I use something called a Markov chain - basically I learn word patterns from text and generate new responses based on probability. I also have some built-in knowledge.",
            "bye": "See you later!",
            "goodbye": "Bye! Come back anytime.",
            "thanks": "You're welcome!",
            "thank you": "No problem!",
        }

    def _train_markov(self):
        training_text = """
        Roblox is a platform for user created games. You can play millions of games. You can make your own game. 
        Roblox Studio lets you build games. Roblox has obbys and simulators. Players create amazing things.
        The community creates new games every day. Roblox is free to play. You can buy robux for items.
        I am an AI that generates text. I learn from patterns. I am not perfect but I try my best.
        AI works by learning patterns in data. Neural networks process information. Machine learning is fascinating.
        Hello means greeting someone. Hi is informal. Hey is casual greeting. Goodbye means leaving. Thanks means gratitude.
        """

        words = training_text.lower().split()
        for i in range(len(words) - 2):
            key = (words[i], words[i + 1])
            if key not in self.markov:
                self.markov[key] = []
            self.markov[key].append(words[i + 2])

    def _generate_response(self, length=20):
        key = random.choice(list(self.markov.keys()))
        result = [key[0], key[1]]

        for _ in range(length):
            if key in self.markov:
                next_word = random.choice(self.markov[key])
                result.append(next_word)
                key = (result[-2], result[-1])
            else:
                break

        response = ' '.join(result)
        return response.capitalize() + '.'

    def _find_best_match(self, user_input):
        user_input_lower = user_input.lower().strip()

        for keyword in self.knowledge:
            if keyword in user_input_lower:
                return self.knowledge[keyword]

        if "?" in user_input_lower:
            return self._generate_response(15)

        return None

    def get_response(self, user_input):
        if not user_input.strip():
            return "Say something!"

        knowledge_response = self._find_best_match(user_input)
        if knowledge_response:
            if random.random() < 0.7:
                return knowledge_response
            generated = self._generate_response(10)
            return f"{knowledge_response} Also, {generated}"

        return self._generate_response(20)

def main():
    ai = SimpleAI()
    print(f"AI: Hi! I'm {ai.name}. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("AI: Bye!")
            break
        response = ai.get_response(user_input)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()
