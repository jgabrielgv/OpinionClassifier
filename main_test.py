import main as s

print("Processing csv file...")
print(s.sentiment_file("Test.csv"))

print("Testing words")
print(s.sentiment("Demasiado buena la peli :-))"))
print(s.sentiment("Mala mala demasiado mala"))
print(s.sentiment("duran"))
