# WeEvolve Launch Demo — Terminal Recording
# Run: vhs demo.tape
# Output: demo_terminal.mp4

Output demo_terminal.mp4
Output demo_terminal.gif

Set Shell "zsh"
Set FontFamily "SF Mono"
Set FontSize 18
Set Width 1200
Set Height 800
Set Padding 40
Set Framerate 60
Set TypingSpeed 50ms

# Scene 1: The Install
Sleep 500ms
Type "pip install weevolve"
Enter
Sleep 3s

# Scene 2: First Run
Type "weevolve"
Enter
Sleep 5s

# Scene 3: Learn Something
Type "weevolve learn --text 'Agent orchestration is the real moat. Not models, not prompts — coordination wins.'"
Enter
Sleep 4s

# Scene 4: Check Status (shows level up)
Type "weevolve status"
Enter
Sleep 5s

# Scene 5: Recall Knowledge
Type "weevolve recall 'agent orchestration'"
Enter
Sleep 4s

# Scene 6: Evolve
Type "weevolve evolve"
Enter
Sleep 4s
