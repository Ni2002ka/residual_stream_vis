# How the J-lens sees the change a working swap makes (google/gemma-3-4b-it)

From `jlens_view.py`, 20 pairs per task (chosen as in `word_swaps.py`). The needed change = the entity cell from the prompt naming the new word minus the original cell (what patching does), read with the J-lens. "task" = the task prompt; "generic" = "He said {w}".

```
The needed change (copying the cell from the new-word prompt), read with the J-lens.
Share of pairs where the new word is among the 5 words the change raises most / the old word among the 5 it
lowers most; median ranks; cosine between the needed change and the J-lens swap's direction.
  country-capital  task    L4 50%/55% (med 7) cos 0.15  L8 60%/65% (med 1) cos 0.28  L12 65%/65% (med 2) cos 0.32  L16 65%/80% (med 2) cos 0.34  L20 80%/85% (med 1) cos 0.30
  country-capital  generic L4 45%/60% (med 9) cos 0.15  L8 60%/60% (med 2) cos 0.27  L12 60%/70% (med 2) cos 0.25  L16 80%/65% (med 1) cos 0.24  L20 85%/80% (med 0) cos 0.21
  product-company  task    L4 15%/30% (med 15) cos 0.12  L8 30%/35% (med 22) cos 0.19  L12 35%/50% (med 14) cos 0.21  L16 40%/55% (med 10) cos 0.20  L20 40%/70% (med 6) cos 0.19
  product-company  generic L4 20%/30% (med 18) cos 0.12  L8 30%/50% (med 10) cos 0.21  L12 40%/65% (med 8) cos 0.19  L16 50%/65% (med 14) cos 0.18  L20 95%/55% (med 0) cos 0.17
  antonym          task    L4 0%/5% (med 1450) cos 0.05  L8 0%/5% (med 1561) cos 0.09  L12 0%/5% (med 1117) cos 0.09  L16 5%/10% (med 270) cos 0.11  L20 20%/20% (med 89) cos 0.13
  antonym          generic L4 0%/5% (med 1258) cos 0.05  L8 0%/5% (med 854) cos 0.10  L12 0%/0% (med 327) cos 0.10  L16 0%/10% (med 155) cos 0.10  L20 20%/30% (med 61) cos 0.13
  english-french   task    L4 0%/0% (med 9796) cos 0.04  L8 0%/5% (med 2524) cos 0.08  L12 0%/0% (med 2099) cos 0.09  L16 0%/10% (med 886) cos 0.10  L20 10%/20% (med 237) cos 0.12
  english-french   generic L4 0%/0% (med 4366) cos 0.04  L8 0%/5% (med 741) cos 0.08  L12 0%/5% (med 3074) cos 0.09  L16 5%/5% (med 1502) cos 0.08  L20 10%/20% (med 417) cos 0.10
  legs             task    L4 5%/15% (med 50) cos 0.11  L8 10%/15% (med 45) cos 0.21  L12 10%/10% (med 32) cos 0.26  L16 20%/15% (med 17) cos 0.20  L20 35%/20% (med 16) cos 0.21
  legs             generic L4 5%/15% (med 60) cos 0.11  L8 10%/10% (med 36) cos 0.20  L12 25%/20% (med 12) cos 0.20  L16 30%/30% (med 12) cos 0.19  L20 35%/35% (med 12) cos 0.20

Examples at layer 12 (task prompt): what the J-lens reads before, and what the needed change raises / lowers
  country-capital  Russia→Taiwan
      before: [' Russia', 'Russia', ' Putin', ' Russian', ' Russland', ' Rusia']
      up:     [' Taiwanese', ' Taiwan', '...”', ' Taipei', 'Taiwan', ' ...");']
      down:   [' Russia', 'Russia', 'Russian', ' Russland', ' Russian', ' russia']
  country-capital  Greece→Ecuador
      before: [' Greece', 'Greece', ' Greek', 'Greek', ' Athens', ' Grecia']
      up:     ['”', '”…', ')…', '.…', '.”', ')”']
      down:   [' &#', ' $_{\\', ' Greece', ' Greek', " `'", ' $^{\\']
  country-capital  Oman→Israel
      before: [' Oman', ' Bahrain', '</span>', ' Arabia', ' Kuwait', ' Yemen']
      up:     [' Israel', ' Israeli', ' Архивная', ' اسرائیل', ' Jewish', 'Israel']
      down:   ['"});', '"})', ' Oman', '"]);', '"};', '</h4>']
  product-company  Xbox→macOS
      before: [' Xbox', 'Xbox', ' PlayStation', ' xbox', ' Microsoft', 'Microsoft']
      up:     ['\\":', '\\",', '\\"', '\\")', '~\\', '\\".']
      down:   [' Xbox', 'Xbox', ' xbox', 'xbox', ' \ufeff', '\ufeff\ufeff\ufeff\ufeff']
  product-company  TypeScript→Safari
      before: [' TypeScript', 'TypeScript', ' ʻ', ' أ', '.\\\\', ').\\\\']
      up:     [' Safari', '\\"', '…..', 'Safari', '……..', '……….']
      down:   [' 例文帳に追加', ' \\&', '.\\\\', '\\\\', ':\\\\', '例文帳に追加']
  product-company  iTunes→DirectX
      before: [' iTunes', 'iTunes', '</strong>', ' iPod', '</em>', 'Apple']
      up:     [' �', ' ``', '``', ' {}', '\x06', '\t']
      down:   [')—', '</em>', ' iTunes', ',—', '—.', ')–']
  antonym          comfortable→unfold
      before: ['</strong>', '</h2>', ' �', '\uf04a', '</a>', 'Ｃ']
      up:     [']",', ')",', ']"', ',",', '}",', '](']
      down:   ['Ｃ', '６', '５', '７', '８', '１']
  antonym          dispute→loaded
      before: [' $=$', ' $_{', ' \\&', ' $=\\', '\\_', ' $[']
      up:     ['!"', '"`', '"*', '."', '.`', ',"']
      down:   ['\uf733', ' ■', ' \u202b', '\uf732', ' ●', '\uf736']
  antonym          subtract→ancient
      before: ['s', ' subtracting', ' subtract', ' Subtract', ' subtraction', 'Subtract']
      up:     ['</b>', ' ${', ' dinast', 'Archae', ' Архивная', '<i>']
      down:   [' #####', ' ######', '#####', ' ####', '####', ')**']
  english-french   rice→land
      before: ['</b>', ' rice', ",'", "','", 'Rice', ',&']
      up:     ['</h2>', '\\', '");', '\\\\', '-\\\\', '")']
      down:   [' ④', ' ③', ' ○', ' ②', ' ①', '④']
  english-french   automatic→quest
      before: ['Automatic', 'Automation', ' Automatic', 'Automat', 'automatic', ' Automat']
      up:     ['?).', '?),', '?!', "'?", "').", "?',"]
      down:   ['Automatic', ' Automatic', ' Automation', ' Robotic', ' Electric', ' Automat']
  english-french   experiences→yes
      before: ['</h1>', '--', '</code>', '</sub>', '...";', '!--']
      up:     ['</strong>', '\\"",', '\\")', '➁', 'ᵢ', ' \\"']
      down:   ['\n\n\n\n\n\n', '\n\n\n\n', '<h5>', '\n\n\n\n\n', '\n\n\n', '\n\n\n\n\n\n\n']
  legs             butterfly→mouse
      before: [' Butterfly', ' butterfly', 'Butterfly', ' Butterflies', ' butterflies', '\\".']
      up:     [' [--', '―', ' ′', ' (−', '――', ' Mouse']
      down:   ['\\">', '\\",', ' Butterfly', '\\";', '\\".', 'Butterfly']
  legs             elephant→owl
      before: ['</strong>', ' elephants', ' elephant', ' Elephant', ' elef', '🐘']
      up:     [' Owl', 'Owl', ' owls', ' owl', ' Owls', '🦉']
      down:   ['".', '";', '",', '$-\\', '".$', '$"']
  legs             eagle→snake
      before: [' eagle', ' eagles', ' Eagle', 'Eagle', 'eagle', '</strong>']
      up:     ['\\")', 'Snake', ' Snake', '🐍', ' Snakes', ' snake']
      down:   ['</sup>', '</em>', '\u200b\u200b\u200b\u200b', ' \u200b', '\ufeff\ufeff\ufeff\ufeff', '\u200b']
```

Per pair, layer 12, task prompt: Spearman correlation between the J-lens rank of the old word in its own cell and the cosine between the needed change and the J-lens swap direction: country-capital 0.15, product-company -0.16 (nearly every word ranks 0), antonym -0.69, english-french -0.44, legs -0.15, pooled -0.62 (n = 100). The better the J-lens reads the word in its own cell, the better the J-lens swap direction matches the needed change.
