# UE5 Project Setup
1. Create new UE5.4 C++ project in /ue_project
2. Enable plugins: USD Importer, Live Link, nDisplay, RTX
3. Enable Python: Edit -> Plugins -> Python
4. Content structure:
   /Content/Humanoid/ - MetaHuman or H1 mesh
   /Content/USD/ - Live USD stage from /tmp/humanoid_live.usda
5. Level Blueprint:
   - On Tick, reload USD stage
   - Add debug line traces for contact forces (from causal graph JSON)
6. For AAA look:
   - Use Lumen GI + Nanite meshes for environment
   - Megascans for floor / lab environment
   - Post Process: Motion blur off (for training clarity)
