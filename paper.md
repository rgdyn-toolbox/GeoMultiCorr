# GeoMultiCorr
## Introduction
### Scientific background
- Growing needs Earth Surface displacement data
	- RGV (TD)
	- Landslides (DC)
- The existing tools
	- ImCorr
	- CIAS
	- **Major limits** 
		- not possible to pre-process, process and post-process automatically the images 
			- This is limitating the usages on very small regions or short timespans 
		- Not possible to add new algorithms
		- Community not large and not very active
	- What about ArrowSix ?
### Objectives
- Propose to the community a toolbox to 
	- Manage a large images database
	- Generate optical displacement fields (ODF) from the database
	- Post process the ODF
		- Evaluate the quality
		- Identify and correct the bias
	- Analyse the ODF
		- We can say "provide basic analysis tools" but insist on the fact than geoscientists have to then produce their own analysing code (for the rgs it would be the moving areas detection for instance, and the derivation of RGVs) 
		- And like this, we clearly make a difference between the tool and its application to answer scientific questions
- Make a tool Usable globally on all type of optical images (satellites, aerial)
- Open Source
## Method
### The core : Ames Stereo Pipeline
- Why ASP and not another images correlator ?
	- Very active community
	- Highly customizable
	- Highly performant and robust
		- The algorithms don't reduce the images resolution for instance, at contrario from CIAS or ImCORR
### The programm structure : object oriented philosophy
- Project
	- Composed of the database and the management scripts 
- Session
	- Controlling Center. 
	- One session is initialized at each time we connect to the database to execute queries or runs. On session is related to a project.
- Pair
	- One pair is one run, basically.
	- One pair belongs to only one pzone.
	- One pair have two thumbs.
- Thumb
	- There is two types of thumbs
		- the natives
			- RGB images at full resolution cropped on the pzones
			- One native thumb belongs to one Pzone.
		- the pair-relateds
			- Here one thumb belongs to **one pair**.
			- Only one band has been extracted and eventually resampled to a coarser resolution than the native.
			- 2 pair related thumbs are ready to send to ASP stereo corr
- Pzone
	- One Pzone can have many pairs.
## Results
### Short examples of applications
- I think the results are not only the disp fields itselves but also the database and its management possibilities
- Examples of disp fields computed in different part of the world and on different landforms (Rock Glaciers, Landslides)
### Scientific applications (very rough examples of potential paper titles which beyond the scope of this paper)
-  *Controlling factors of Rock Glaciers surface displacements in Switzerland (Thibaut)* 
- *Temporal variation of rock glacier velocities in the Andes... (Diego) ?*