# Gurdy: Voice-Controlled Hexapod Robot in Hindi

🎥 **Demo video:** _[add YouTube link here]_ · Code: [README.md](README.md)

## Problem
Robots are controlled through English software, keyboards and apps. Farmers, field workers and many other people in India would find it easier to tell a robot what to do in their own language.

## Solution
Gurdy is a six-legged robot (simulated in Gazebo) that you control by speaking Hindi or Hinglish. It walks, describes what its camera sees, walks up to objects, and answers aloud. Every reply starts with "Main ek AI assistant hoon", so users know they are talking to an AI.

## How it works
- **Speech to text:** Gnani **Prisma v3.0** turns the Hindi speech into text.
- **Understanding:** a keyword parser (Devanagari and romanized Hindi) turns the text into a JSON command, e.g. `{"action": "walk_forward", "distance_m": 0.2}`.
- **Robot:** ROS Noetic + Gazebo 11 in Docker on Windows/WSL. Scripted gaits walk in **4 directions** (forward, backward, left, right) with 0 falls in testing.
- **Vision:** **OpenCV** HSV colour detection finds the red ball and blue/green boxes, with direction, rough distance and shape. *Go to object* side-steps until the object is centred, then walks up to it.
- **Reinforcement learning:** a **DQN** trained for **450 episodes** controls each leg (25 actions). Checkpoint 450 walks **14.5 cm forward on average with 0/3 falls**. In **RL mode** the robot walks with the DQN and switches to the **scripted gait** if the policy stalls.
- **Text to speech:** Gnani **Timbre v2.5** (voice Poorvi) speaks the reply in Hinglish.
- **Web dashboard** (localhost:8080): map, live camera with detections, status, and Speak / Stop buttons.

## Real-world uses
Voice control in Indian languages for field robots in agriculture, inspection, and search and rescue, and for assistive devices, so people can give commands without English software.

## Honest limitations
- Simulation only, not yet on real hardware.
- Gurdy cannot turn (its legs move only along radial spokes), and sideways and backward walks drift a few degrees.
- On flat ground the scripted gait is still better than RL. RL stalls after about 15 cm, so the fallback is needed.
- Commands are keyword-based. Misheard words may need repeating.

## Next steps
- Train RL where fixed gaits fail: slopes, rough or slippery ground, a damaged leg.
- An RL navigator that picks gait strokes to reach a target and correct drift.
- More languages (Kannada, Tamil…) and a real hexapod.
