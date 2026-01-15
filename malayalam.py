import pandas as pd
import random

# -----------------------------
# DEFINE MALAYALAM SENTENCES DIRECTLY
# -----------------------------
malayalam_sentences = [
    "ഞാൻ ഇന്ന് സ്കൂളിലേക്ക് പോകുന്നു.",
    "എന്റെ സുഹൃത്തുക്കൾക്കൊപ്പം പുസ്തകം വായിച്ചു.",
    "അവർ വളരെ നന്നായി പഠിക്കുന്നു.",
    "ഞാൻ ഫലം കഴിച്ചു.",
    "അവൾ വീട്ടിൽ ഉറങ്ങി.",
    "ഞാൻ ഇന്ന് മാർക്കറ്റ് പോകും.",
    "അവൻ നല്ല പാഠം പഠിച്ചു.",
    "ഞാൻ എന്റെ സുഹൃത്തിന്റെ വീടിലേക്ക് പോകുന്നു.",
    "അവൾ പാഠം എഴുതുന്നു.",
    "ഞങ്ങൾ സിനിമ കാണാൻ പോവുന്നു.",
    "ഞാൻ രാവിലെ ഓട്ടം ചെയ്യും.",
    "പക്ഷികൾ ആകാശത്തിൽ പറക്കും.",
    "അവൻ പുസ്തകം വായിച്ചു.",
    "ഞാൻ പുതിയ വസ്ത്രം വാങ്ങി.",
    "അവൾ പാചകം ചെയ്യുന്നു.",
    "ഞങ്ങൾ പാരക്കുട്ടി കളിക്കുന്നു.",
    "ഞാൻ എന്റെ കൈ കഴുകി.",
    "അവൻ വീട്ടിലേക്ക് വരുന്നു.",
    "ഞാൻ പഴം കഴിക്കുന്നു.",
    "അവൾ സ്നേഹത്തോടെ പാഠം പഠിക്കുന്നു.",
    "ഞാൻ കഫേയിൽ പോകുന്നു.",
    "അവൻ എന്റെ സുഹൃത്തെ കണ്ടു.",
    "ഞാൻ പുസ്തകം തുറന്നു.",
    "അവൾ കവിത വായിക്കുന്നു.",
    "ഞങ്ങൾ പുഞ്ചിരിയോടെ സംസാരിക്കുന്നു.",
    "ഞാൻ റോഡിൽ നടക്കുന്നു.",
    "അവൻ ഗാനം പാടുന്നു.",
    "ഞാൻ ബസ് പിടിക്കും.",
    "അവൾ സദ്യ കഴിക്കുന്നു.",
    "ഞാൻ മഴയിൽ നടന്നു.",
    "അവൻ സിനിമ കാണുന്നു.",
    "ഞാൻ ഫോട്ടോ എടുത്തു.",
    "അവൾ വീടിനു പോകുന്നു.",
    "ഞങ്ങൾ ബസ് സ്റ്റേഷനിൽ എത്തി.",
    "ഞാൻ പുതിയ പുസ്തകം വാങ്ങി.",
    "അവൻ ജോലിയിൽ തിരക്കിലാണ്.",
    "ഞാൻ പാചകം പഠിക്കുന്നു.",
    "അവൾ പൂവുകൾ നിരത്തി.",
    "ഞങ്ങൾ പാർക്കിൽ കളിച്ചു.",
    "ഞാൻ സംഗീതം കേൾക്കുന്നു.",
    "അവൻ പുതിയ വസ്ത്രം ധരിച്ചു.",
    "ഞാൻ പെയ്ന്റ്റിംഗ് ചെയ്യുന്നു.",
    "അവൾ പുസ്തകം വായിക്കുന്നു."
]

# -----------------------------
# AUGMENTATION FUNCTION
# -----------------------------
def augment_sentence(sentence):
    punctuations = ['.', '!', '?', '…', '']
    words = sentence.split()
    if len(words) > 2 and random.random() > 0.7:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    sentence_aug = ' '.join(words)
    sentence_aug += random.choice(punctuations)
    return sentence_aug

# -----------------------------
# GENERATE AUGMENTED DATASET
# -----------------------------
augmented_sentences = []
target_count = 3000
while len(augmented_sentences) < target_count:
    orig = random.choice(malayalam_sentences)
    augmented = augment_sentence(orig)
    augmented_sentences.append(augmented)

# Create DataFrame
df_augmented = pd.DataFrame({
    'sentence': augmented_sentences,
    'language': ['Malayalam'] * len(augmented_sentences)
})

# -----------------------------
# SAVE TO CSV
# -----------------------------
output_path = r"C:\Users\hp\Downloads\malayalam_augmented.csv"
df_augmented.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Augmented dataset saved to {output_path}")
print("Total sentences:", len(df_augmented))



