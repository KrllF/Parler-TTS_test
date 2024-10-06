import whisper
import jiwer
import numpy as np
import pandas as pd
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_parler = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-jenny-30H").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-jenny-30H")
model_whisper = whisper.load_model("large")

references = ["I think the fact that the planet Earth is shaped like a ball is irrefutable.",
"If you can buy some kinds of fruit and yoghurt, I'll make a delicious salad."
"Galapagos tortoises are extraordinary animals that can live for hundreds of years."]
predictions = []
def wer_help():
    description = "Jenny speak very fast, quite clear, very confined sounding, quite monotone"
    for prompt in references:
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generation = model_parler.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
        result = model_whisper.transcribe("parler_tts_out.wav")
        predictions.append(result['text'])

wer_help()

wer = jiwer.wer(references, predictions)
print(f"WER: {wer}")


to_much = ["Microscopic particles of plastic in the ocean contribute not only to water pollution but also to the extinction of its inhabitants.",
"I don't know a single person who doesn't like cats, as they are very cute and beautiful.",
"Did you know that the Earth makes a complete revolution around its axis in 23 hours 56 minutes and 4 seconds?",
"A millisecond is a unit of time lasting one hundredth of a second.",
"St Petersburg is the northernmost of the world's major metropolises and is sometimes called the City of White Nights."
"St. Petersburg is 317 years old, but the oldest monuments of the city are the Egyptian sphinxes on the University Embankment, their age is about 3500 years.",
"A constantly changing organic network with infinitely complex and dynamic interrelationships, giving unexpected sprouts - that's what a proper modern organisation is.",
"If I were now allowed to choose any animal I wanted to get - it would certainly be a cat.",
"As a rule, ordinary people favour milk chocolate, but I think white chocolate is much tastier.",
"My grandfather had an unusual hobby when he was young: he used to breed pigeons.",
"Lions are among the most majestic animals on Earth, they are called the kings of the jungle.",
"The human brain weighs about 1.4 kilograms on average, but it consumes about 20% of the body's total energy.",
"If you're ever in Paris, make sure you climb the Eiffel Tower - the view is amazing.",
"Dolphins have the ability to sleep with only one half of their brain to stay alert.",
"Scientific studies show that regular exercise improves mood and reduces stress levels.",
"Sports and proper nutrition are key to maintaining a healthy lifestyle.",
"Turtles don't make sounds, but they can hiss when they feel threatened.",
"Scientists believe there may be life forms in space that are not carbon-based like on Earth.",
"Don't forget that regular sleep helps you maintain good health and high productivity.",
"In the last hundred years, there has been a huge amount of technology invented in the world that has changed the lives of mankind.",
"Penguins are amazing birds that can't fly, but they are excellent swimmers.",
"Water is the basis of all life on Earth, without it living organisms cannot exist.",
"Bats are the only mammals that can fly.",
"Ants can lift objects that are several times their own weight.",
"The Sahara is the largest desert on Earth, but it was once home to forests and animals.",
"The ozone layer protects our planet from the sun's harmful ultraviolet radiation.",
"Caffeine helps to temporarily boost concentration, but excessive use can lead to insomnia.",
"The largest flower in the world is the Rafflesia arnoldi, which can reach a diameter of up to 1 metre.",
"Interesting fact: octopuses have three hearts, one of which helps pump blood through their gills.",
"About 70% of the Earth's surface is covered with water, and only 3% of that amount is fresh water.",
"If you want to learn to play the piano, don't put it off, because the sooner you start, the sooner you will succeed.",
"Many ancient civilisations built their cities near rivers because water was the most important resource."]
