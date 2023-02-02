# For Set1
# dims_to_dist = {
#     "content": [125, 512, 1108, 1119, 608, 111],
#     "organization": [157, 580, 1198, 1061, 535, 52],
#     "word_choice": [147, 535, 1285, 1069, 469, 78],
#     "sentence_fluency": [113, 373, 1227, 1249, 554, 67],
#     "conventions": [145, 501, 1194, 1188, 508, 47]
# }

# For Set2
dims_to_dist = {
    "content": [1268, 2011, 2488, 1161, 173],
    "prompt_adherence": [1003, 1775, 2776, 1373, 174],
    "language": [1038, 1843, 2862, 1205, 153],
    "narrativity": [1248, 1882, 2642, 1162, 167],
}

freqs = [0] * len(dims_to_dist["content"])
for dim, dist in dims_to_dist.items():
    print(dim)
    for i, freq in enumerate(dist):
        # print(round((sum(dist) - freq) / sum(dist), 2))
        freqs[i] += freq

for i, freq in enumerate(freqs):
    print(round((sum(freqs) - freq) / sum(freqs), 2))
