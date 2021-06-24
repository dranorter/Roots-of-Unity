
I'm a big fan of the Catalan solids, which are way cooler than their dual, the Archimedian solids, because Catalan solids make fair dice. Their faces are all identical, and because their vertex figures are regular, the corners each roll one direction as easily as any other.

My favorite Catalan solid is the rhombic dodecahedron, which makes a nice 12 sided die, but also has the nifty property of filling space. So I originally thought I would make a Minecraft clone which used rhombic dodecahedra instead of cubes. 

*Onscreen text: There used to be a mockup of this online somewhere, but I can't find it anymore.*

But I like to complicate things, as you may have already noticed.

*Onscreen text: Laser-cut something-or-other, custom seximal notation*

So instead I decided to make a voxel game using an aperiodic tiling.

The prototypical example of an aperiodic tiling is the Penrose tiling, although there are others which are quite beautiful.
(0:50)
I knew that the Penrose tiling could be made 3D, because I'd seen some photos of a way of presenting the tiling as 3D, just based on projecting the parts of the 2-dimensional Penrose tiling outwards.

So the Penrose tiling itself is based on a 5-dimensional grid which is projected down to 2 dimensions in a very symmetrical way, although you do get things which are similar to the Penrose tiling if you project them down in a non-symmetrical way. But part of the fascination of the Penrose tiling is that it's only using 2 tiles, and you acheive that by projecting down the 5 axes symmetrically.

So, you don't have to think about 5 dimensions in order to accomplish this - the original structure was referred to as a Multi-Grid; I don't know if Ammann was the one who made up that term, or DeBruijn or Penrose; one of those early pioneers. A multi-grid is just equally spaced lines rotated and overlayed on one another. The resulting mess is used to create the Penrose tiling. But you can think of the 5 directions in the multi-grid as being projected down from 5D, and then use some of the 5-dimensional information that remains after the projection to pull some of these vertices up into a quite nice 3 dimensional structure where instead of using 2 tiles, you're just using one tile, called the golden rhombus.

And so, having seen photos of this without having looked at the details, I just thought it would be straightforward to do the 3 dimensional thing based on the 5 dimensional thing; in other words, take a 5 dimensional cubic lattice and project it straight to 3D. This turned out to be not quite how it works, but not too far off either.
(3:30; so 4:20)
5-dimensional cubes can't be projected down to 3D in a fully symmetrical way, but 6-dimensional cubes can, and the result fills 3 dimensional space with two rhombohedra. The rhombohedra are called the "golden rhombohedra" or the "Ammann rhombohedra", and the tiling is knows as either the "3D Penrose tiling" or the "3D Ammann tiling" (Ammann originally proposed it). Like the Penrose tiling, it uses just two tiles, and like the Penrose tiling, there's a way of "decorating" the tiles which prevents them from making a periodic tiling, and instead forces the 3D Penrose tiling.

However, the 2D Penrose tiling has an important property which this 3D version tiling is missing. The 2D Penrose tiling has a very strict "hierarchical self-similarity", meaning it has "inflation"-slash-"deflation" rules. This allows any Penrose tiling to be subdivided into a new, smaller Penrose tiling, or grouped together to form a new, larger Penrose tiling. This doesn't quite work for the 3D Penrose tiling. This problem, along with other more technical issues, has caused the 3D Penrose tiling to be often set aside in the study of quasicrystals. Paul Steinhardt, one of the pioneers of quasicrystal research, has always favored a closely related aperiodic tiling which I'll call the Steinhardt tiling.

The Steinhardt tiling is composed of 4 shapes, each of which can be easily built within the 3D Penrose tiling. The first shape is one of the two golden rhombohedra; the second is the Bilinksi dodecahedron, which is a squished version of my friend the rhombic dodecahedron. The third is a 20-sided lens shape, and the largest tile is a 30-sided shape called the Rhombic Triacontahedron - a Catalan Solid!

All 4 of these can be sliced up in a way which creates a smaller Steinhardt tiling, which means the Steinhardt tiling has valid inflation and deflation rules.

Inflation and deflation are very appealing to me, because in a voxel game, they would mean I can group "blocks" together into "chunks", and then group those "chunks" together into larger groups, or "regions". This makes it possible to save and load small segments of the world as the player travels around it, rather than trying to manage everything all at once.

So initially I planned on making a voxel game using the Steinhardt tiling, so that I could have elegant chunks based on the tiling itself. But it turned out the 3D Penrose tiling, ie, the Ammann tiling, was easier to program, so I ended up examining it first. I had Paul Steinhardt's word that chunks would be impossible in this tiling, but compared to the Ammann tiling, the Steinhardt tiling is very blocky; the Ammann tiling is just the Steinhardt tiling except we're allowed to take apart the big tiles into golden rhombohedra.

So I agonized for a bit. My most optimistic side just wanted a game which would allow the user to select any tiling they wanted, including periodic tilings like cubes or rhombic dodecadedra, alongside the 3D Penrose tiling, Steinhardt tiling, and other aperiodic tilings like the Danzer tiling. But if I wanted cohesive gameplay and aesthetics I knew it would have to be based on a single tiling, and the 3D Penrose Tiling won out. I just feel like there are more interesting things which can be built in it than in the Steinhardt tiling.

Having decided this, I knew that I would have to prove Steinhardt wrong, and find a way to generate the tiling with inflation rules.

Cutting a long story very short, after months of programming and math-ing, I'm successfully doing inflation and deflation, which means I can generate as much of the 3D Penrose tiling as I want. But instead of using just 2 rules like the 2D Penrose, my method requires 4980 rules; and I have to preserve information from the 6D cubic lattice to decide when to use which rule. But basically, I'm finally where a cube-based Minecraft clone would be on day 1. So I figured it was time to make a devlog about it.

Meanwhile, my concept of the game I'm creating has evolved. So here's my over-ambitious feature list.

1) Since I'm preserving the full 6-dimensional location of every block, I'd like to make that available in gameplay somehow. The mechanics I've come up with require a lot of explanation, but basically there should be a way to slowly build out into 6 dimensions, or bridge through 6D to a nearby 3D world, that sits at a different "angle" within 6D. I like the idea of making a mechanic which people could use, without understanding the math.

But here's a bit of an explanation. From the point of view of the 6D space, the 3D world is roughly flat, and not aligned to any one of the 6 dimension axes. This world-plane is then corrugated, like a slope in Minecraft, since every voxel lines up exactly with 3 of the 6 dimensions.

So what I'm thinking is there'd be a material which you can place out in the world, surrounding one of these points where the world is corrugated; and then you could hit it, and the corrugation would flip, sort of taking this flat 3-dimensional world and bending it outward slightly into 6-dimensional space. And if you bend it far enough, it "breaks" and lets you walk off the world-grid entirely.

2) One of the first things I want to add to the game is good rain mechanics, with puddles and flowing water. The aesthetic of the game will be very rainy and desolate, at least at first.
(timestamp unknown)

3) I want to have a spherical planet. I'm still deciding what's going to make the game look best locally, so I'm still deciding whether I'm going to have a rhombic triacontahedron planet or a dodecahedron planet, and then the gravity on each face will just point inwards, perpindicular to the face, so that you can have some consistency when placing blocks, that there will be a part that's pointing upwards - at least consistent enough to have some decent art for the blocks, like having grass on top of the blocks, that sort of thing. I want to have the chunk generation be nice enough that you could fall from space and land on the surface, and have the terrain generate in as you fall; like you could see the planet, from a distance just looking like the rhombic triacontahedron, and then as you get closer it starts to generate some surface details. It might decide that a certain chunk is going to generate as ocean and color it as water, and then as you get closer and closer to the ground, it will generate smaller and smaller features until you get to block level and you land.
(3:00)

4) Since I can "inflate" the lattice to go from blocks to chunks, I can also "deflate" blocks to get sub-blocks. I think it'd be nice to allow "chiseling" blocks into sub-blocks, but I have no idea if that'll look good, because some of the blocks are going to go off the sides. So you'd chisel this nice parallelogram, and it would just become this horrible pointed mess, whichever sub-blocks I'd assign as directly belonging to that block. Still, it's worth a try.

5) I'd like to have the variety of biomes be somewhere near as creative and challenging as the game Hyperrogue. Basically, Hyperrogue biomes each have mechanics that affect gameplay, rather than just looking different. That's a very different sort of game, so that's a tall order for sure, but I like the feeling of exploring in that sort of world. 

6) I feel like world generation is one of the more fun parts to write with these sorts of games, and I'd like to have the world generation be sort of accessible to the player as well. What I'm currently envisioning is a system where generated features of the terrain have "spirits" which are in control of the feature; so there'd be mountain spirits and river spirits. Maybe every chunk would have a spirit. Generated buildings would have spirits, and then the spirits of buildings would be the most straightforward spirits. They'd literally be the inhabitants of the building. And you can trade with them and if you give them fancy materials, they might start making the building out of those materials. Other spirits like river spirits you could try to take control of in order to try and move the river.

7) Speaking of taking control, I want there to be a rich ecosystem with different animals, and one of the ways the player would progress is by taking control of more and more powerful animals. I've always liked games like Abe's Oddessey, or even just the Zerg in Starcraft, where you can take control of enemy units. Ultimately I'd like a mechanic where you'd try and breed the best dragon you can, in order to mind-control it and fly around.

8) I'd like to have a lot of long-term physics, with dead wood slowly losing its color, spider webs appearing in indoor areas, and forest fires spreading even when the area on fire isn't fully loaded. Optimistically, plants and animals will thrive or disappear in the player's absence too, and the player could try to prevent or cause environmental disasters.

Development so far has been pretty interesting and challenging, and this video would be three hours long if I tried to explain everything. What I'm going to try to do with these devlogs going forward is mix in some of those explanations with progress updates.

Alright, thanks for listening! Next video I'll aim to have a creative mode working and some very basic terrain generation.