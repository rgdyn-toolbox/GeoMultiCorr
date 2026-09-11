# Compte-rendu
## Github repository
- TD presented to DC the actions listed in [[2026-09-11_online_TD-DC]]
- We now have a unique repository with 3 branches
	- DC's version of the GeoMultiCorr software 
	- TD's version
	- paper
## Publication strategy
- TD and DC both agree on Joss as an appropriate journal to publish our software GeoMultiCorr
## Development strategy
### Version conflicts
- The code of GeoMultiCorr is currently existing in both versions which are very different one to the other. Each of it have its own logic, pros, and cons,
- Our **principal objective** in order to publish the software is to merge these versions into one. In this order, TD have created the note [[issues]]. 
### GeoMultiCorr project's scope
- Should we integrate the part which download the images in GeoMultiCorr or not ? 
	- DC spent a lot of time on a module to download spot6 images, sentinel... 
	- But it's quite complex to use. 
	- We don't know if we should keep it into GeoMultiCorr or if its to the user to do it on its own side.
## Software Philosophy 
- GeoMultiCorr is very complex. We could imagine its case uses as level-based. 
- Simple usage
	- One displacement field per pzone
	- Simple pair creation strategy
	- No Temporal Inversion
	- Correspond to specific research objectives
- Advanced usage
	- Many displacement fields per pzone
	- Needs to elaborate a complex strategy to create the pair (images apparaiment)
	- The disp fields are then evaluated through both the mono metrics and the multi metrics
	- Time series are generated
- The choice of the level os usage depends on
	- The images we have
	- The research objectives
## TODO
- The first step is to be aware of what we have in hands. Therefore, TD must try the DC version, and vice versa.
- DC needs to commit its last changes
- From the next meeting, we can think about the GeoMultiCorr scope and the questions already open and described in [[issues]] 
---
# TD's action before the meeting
## Paper Plan
- TD's elaborate a rough potential plan for the paper. See [[Intra/GeoMultiCorr/main|main]]
- And a graphical canvas [[graphical_canvas.canvas]] visible with obsidian.
## Publication requirements
- [x] Read infos about Joss journal
- Some important points I found in the requirements :
	- Your paper must not focus on new research results accomplished with the software.
	- Projects developed privately are not eligible until there is a public record of open development: at least six months of public history prior to submission, with evidence of releases, public issues and pull requests
	- Strong positive signal (not a gate, but counts in your favour)
## Code base
- [x] Identify the existing and redondant online repositories
	- https://github.com/rgdyn-toolbox/GeoMultiCorr/
		- branch main
		- <mark style="background: #BBFABBA6;">branch td_side just created</mark> 
	- https://github.com/cusicand/GeoMultiCorr
		- Forked from https://github.com/rgdyn-toolbox/GeoMultiCorr/
			- <mark style="background: #FF5582A6;">Seems not updated : to delete ?</mark>
	- https://github.com/duvanelt?tab=repositories/GeoMultiCorr
		- <mark style="background: #FF5582A6;">Just deleted</mark>
- [x] Gather everything on the same repository
	- I made a git clone from https://github.com/rgdyn-toolbox/GeoMultiCorr/
	- I replaced the .git/ of my own local repo by the one of this repo
	- And created a new branch td_side
		- Then, I made a commit on this branch
- [x] Create a separate branch for the paper, as required by joss
	- The branch paper can host the [[Intra/GeoMultiCorr/main]] file
	- And eventually all the material related to our project management
		- Meeting notes and todos lists
		- Like this, everything is under the same place
## Project strategy
- [x] Represent graphically the conceptual parts of GeoMultiCorr
	- [[graphical_canvas.canvas]]
- [ ] Identify and characterize the ones existing
	-  Only on DC's branch
	- Only on TD's branch
	- Both branches
		- Diverging versions : to merge in a way or another
- [ ] Identify the ones to develop
	- Ex. a module to choose the best algorithms settings on a Pzone