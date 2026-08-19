# L6 desk robot — build spec, constraints and pitfalls

Single source of truth. Supersedes `REVIEW-v4-*.md`, `REVIEW-v5-manifold.md` and the three
`HANDOFF-*.md` files, which have been deleted. Every number here was measured off the built
solids, not read off the source comments.

**Current file: `l6-bot-v54.html` (v5.5).** Earlier versions are kept only as history — do not build
on them. `l6-rocket-*.html` and `l6-sentry-*.html` are separate designs.
---

## 1. What this is

A parametric 3D model (three.js + Manifold, built live in the browser) of a 3D-printed desk
robot: a **Ø130 × 190 mm egg** housing an **NVIDIA Jetson dev kit standing on edge** plus a
**camera module behind an eye visor**. Six printed parts, **no screws and no heat-set
inserts**. The page builds the geometry, renders it assembled or laid out on print plates,
runs a self-audit, and exports STL / 3MF / OBJ / GLB.

Printer: **Bambu P1S, 256 mm bed, 0.4 mm nozzle, 0.2 mm layers.**

### Working in the page

- The audit auto-runs ~1 s after load; a full build + audit takes **~45 s**. Be patient
  before reading `#report` — the main thread is blocked while Manifold works.
- `window.__dbg` holds the live solids: `__dbg.Manifold`, `__dbg.PARTS`, `__dbg.CLASH_SOLIDS`,
  `__dbg.PLATES`. Run your own booleans against these rather than trusting any comment.
- `window.__fit` holds the camera numbers.
- **`PARTS[n].print` is the export boundary.** It is `cleanMesh(forPrint(...))`, and it is the
  only solid the STL writer, the 3MF writer and the audit read. Anything you add downstream of
  the model must read `pt.print`, not `pt.solid`, or it is measuring a different part from the
  one that gets printed.
- The clearance report **derives** its numbers, including every connector's side/top/bottom gap
  through the head window. Figures that appear only in prose have been wrong four times (§7.7).

---

## 2. Hardware — fixed inputs, not negotiable

| item | spec |
|---|---|
| Dev kit | **103 × 90.5 × 35 mm**, stands on edge, bottom face at **z40.4** (top corner z143.4) |
| Camera board | **37 × 37 × 1.4–1.8 mm** (1.6 nominal — measure yours), connectors on the **back** face |
| Lens | Ø24 barrel, **26 mm** long from the board's front face, Ø17 clear aperture, 1.5 mm screw focus |
| Field | **100° diagonal**, assumed 4:3 sensor |
| Connectors | J2 block 14.0 × 10.3 (dx −6.00…+8.40, du 4.40…15.65) · J1 4-pin (dx +3.27…+10.80, du −17.07…−11.60) |
| Heatsink | 26 mm, hangs through the head's 32 × 33.0 window |
| Consumables | **1× zip tie** (camera flex, strain relief), one drop of glue for the visor. That is the entire BOM — the kit's two structural ties went with v5.5's press fit |
### Hard limits on which camera module you may fit

Check with calipers **before** committing — neither is recoverable after the visor is glued.

- **Racked-in barrel projection ≤ 27.0 mm** (nominal 26.0). At 27.5 the barrel hits the
  shell's eye pad during the insertion twist.
- **Nothing over the barrel may exceed Ø26.5** — the visor sleeve is the binding limit: 26.0 +
  2 × 0.25, the worst-case relieved gap with the keel's du float. (Ø26.3 was the old figure,
  derived from a 0.19 mm gap the built solid never had — §5.1d. The clamp admits Ø29.0, so it
  is never the stop.)
- Every 0.1 mm the real barrel is under Ø24 is 0.1 mm back onto the sleeve gap.

---

## 3. Datum table — the numbers everything else follows

Body axis is z. **Nothing above z40.4 may move without re-deriving the board, the lens, the
visor and the eye axis.** That plane is the design's spine.

| datum | value |
|---|---|
| Shell | Ø130 egg, 190 mm tall, rim r53.75, 2.4 mm rim wall |
| Bottom bore | **Ø100 (r50)** — the whole sub-assembly is lifted through it, then twisted 18° to lock |
| Sub-assembly envelope | **≤ r49.5** about the body axis, at every height. The keel is hard-clipped to Ø99 |
| Shell inner radius at the eye pad (z181–185) | **49.0, not 50** — see pitfall 4 |
| Twist-lock lugs | **115° / 215° / 305°**, +18° twist. Spacing 100/90/170° — unequal, so it keys one orientation. Measured: **8 mm wide (9.26°) in a 9 mm channel**, tip r51.85 in a r49.7–52.1 slot, lug z39.9–43.3 in a slot z39.7–43.6 — see §5.1e |
| **Base disc** | r49.5, z36.4–43.6 — the keel's own foot since v5.4 (was a separate floor puck). Two lands stand full height at x < −21.65 and x > 20.75 and carry the lugs |
| **Kit bay** | 42.4 wide (x −21.65…20.75), y ±47.5, floor at **z40.3** — the kit seats on it. Closed at both ends of y, so the disc is a continuous ring |
| **Member root** | every member's bottom face is at **z39.1**, i.e. 1.2 mm *below* the bay floor, so it runs down THROUGH solid base (§7.3). Back plate weld alone: 3.5 × 92 × 1.2 = 386 mm³ |
| Keel back plate | 3.5 mm thick, x −21.65…−18.15, **z39.1–148** |
| Keel tongue | 3.5 × 22, z140–196, behind the camera head |
| Keel front wall | 2.5 mm, x 18.25…20.75, z39.1–57.6 → **36.4 mm shoe interior** for the 35 mm kit |
| Keel end webs | 1.5 mm at y ±46.75 → **92 mm** between them for the 90.5 mm kit. The +y web carries the **power-cable doorway**: 17 wide × z36–56, cut open at the top rather than to a lintel, leaving two ~12.7 mm pillars |
| Aperture | **29.2 × 26.4 capsule** · recess 8.8 mm on the eye axis, 11.65 at the lower rim |
| Connector window | **32 × 33.0**, roof tilted 8.6°, + a 23 × 2.2 local notch for J1 **and an 18.2 × 1.5 local relief for J2** (roof to du 17.5 over dx ±9.1) |
| **Seat frame** | the board's back stop: head plate around the window — **2.5 mm** each side, **2.0** bottom, **2.0 top except 1.0 over dx ±9.1** (the J2 droop relief). Both reliefs are local, so no edge of the frame is ever less than 1.0 mm |
| **Kit seat** | **z43.5** (v5.5) — kit bottom, on four 0.8 mm crush pads over a bay floor at 42.7. Kit top 146.5 against a head-plate face at 146.15 → **0.33 mm of squeeze** on a 35 mm line at y −11.5. Budgeted 20 mm³, measures 16.0. Was z40.4 from v4 to v5.4 |
| **Board tray** | pocket **38.0 wide** (rails inner dx ±19.0), open at the front and the top; ledge top face du −18.6, reaching 1.7 mm forward of the seat plane |
| **Clamp channel** | seat plane → lip underside **5.9 mm**; lips du 19.2…26.0, dx 16.1…19.0, 1.7 thick |
| **Clamp** | 37.0 × 43.6 × 2.8 staple, body **1.3 mm off the board's face**, on two 1.6 × 2.0 mm pads (dx 16.4…18.4, lower end ramped 45° over 1.6 mm) + two snap barbs, crest dx ±19.6 |
| **Latch pockets** | in each rail's inner face: dx 18.8…20.4, du −19.2…−15.3, s S_BB+2.6…6.0 (all in front of the board) |

### Camera fit, measured

- Field **not clipped**: 4.9 mm of margin at the worst (bottom) rim.
- Pupil **does** vignette: **63% bottom / 91% top** at the 100° corners racked out (≈0.7 EV);
  49% / 80% racked in. Unavoidable — no hole in a 33 mm tile passes the whole Ø17 pupil.
- No white is visible from any angle: the charcoal tunnel spans the entire depth over which
  the shell's white bore wall exists, 1.0 mm inside it.
- Board retention: the board slides in **along the eye axis**, 0.5 mm/side in the pocket, 0.1 mm
  drop onto the ledge, and stops on the **seat frame** behind it. The clamp's pads squash it
  **0.3 mm** at 1.6 (0.1 at 1.4, 0.5 at 1.8), camming on over their 45° lead-in, with 0.2 mm of
  clamp float to the lip; the barbs
  deflect 0.6 mm going in, ≈4 N. The plate body clears the front face by **1.3 mm**.
  Sleeve → Ø26 holder **0.40 mm** nominal, **0.25 mm** with the keel at the far end of the
  ±0.5 mm the bore allows (0.15 mm of it lands in du). Both this and the sleeve's own
  **0.80 mm** wall are measured off `bezel` by the audit every run — see §5.1d.

---

## 4. Print plates

| plate | parts | orientation | notes |
|---|---|---|---|
| White | shell | rim-down | no supports. **Brim** — 190 mm tall on a 2.4 mm rim (**755 mm²** measured; the rear cable slot now runs out through the rim and takes 41 mm² of it) |
| Charcoal | keel (base disc + cradle, one part since v5.4) | **upright on its base disc** | **7530 mm²** first layer (measured). Brim anyway — 159.6 mm tall, 102.8 cm³, mass leans forward; the disc reaches 28 mm forward of v5's pan edge, so it is far less of a lever |
| | camera clamp | flat, **pads up** | no overhang — the pads' 45° lead-in rises off the plate in this orientation. Pads down puts 212 mm² of plate 0.6 mm in the air |
| | eye visor | face-up | brim (81 mm²) |
| | antenna | upright | brim (26 mm²) |
| Red | antenna tip | upright | brim |

**The hat and the badge are gone.** They were an alternative crown fitting and its inlay, and
`PARTS` no longer builds either; the black plate went with them. §5 step 9, §7.15 and the
`hat∩antenna` notes below are kept as history. The crown fitting is the antenna, full stop.

**Supports: one patch, one part.** The keel's camera-head underside, ~278 mm² at print z108.
Use a **dense support interface** — it is not a skim patch, it is the foundation of a 40 mm
free-standing pillar (the window's +x side, 7 × 15 mm in section, disconnected from the rest
of the keel from z103 to z143.5). If it detaches, the window roof loses its right-hand anchor.

**Bridges, all anchored both sides, no supports:** shell 1274 mm³ (vents, rear arch, exhaust),
keel 300 mm³, visor 102 mm³. Check the slicer preview for the keel's **32 mm
connector-window roof** — it is a *sloped* bridge (8.6°) and slicers routinely mis-detect
those as overhang and droop them. Droop eats into J2's top clearance, which is **1.8 mm**
after the local relief (§5.1c). The audit computes every connector's side/top/bottom clearance
off the window and its two reliefs and prints the tightest — no hand-carried number.

**No per-part slicer overrides.** Every part prints on the same profile. The visor used to
need one — 0.35 mm line width for a 0.70 mm sleeve — and no longer does: the sleeve is
**0.80 mm = 2 × 0.40** (§5.1d).

---

## 5. Assembly order

The order is load-bearing. Three of these steps were once in the wrong place and each was
provably impossible.

1. **Stand the dev kit on edge in the keel's bay**, ports toward the rear arch, power
   end down. **It goes in ANGLED** — one end down first, then swing the other down; it cannot
   be lowered in flat and never could (§5.1h). **Then press it home**: four crush pads under it,
   the camera head bearing on a 35 mm line across its top. No ties.
   **Leave the barrel plug off.** (v5.4: there is no "slide the keel onto the puck" step any
   more — they are one part. Nothing to align, nothing to pull-test.)
2. **Board into the tray from the FRONT, along the eye axis** — never from above. Feed both
   camera cables back through the head's 32 × 33 window first, then slide the board
   straight back: heatsink and connectors into the window, lens forward, bottom edge onto
   the ledge, back face onto the **seat frame** — that is the back stop. The wedges will hold
   it there once seated; steady it with a finger anyway. Then the clamp: lens through its open bottom, arms down
   the two channels, push until both barbs click into the rail pockets. Its two 2 mm pads
   land on the board's outer border, right over the seat strips; the plate itself bridges
   1.3 mm above everything inboard.
3. **Rack the focus fully IN.**
4. Lift the sub-assembly through the Ø100 bore and **twist 18°** to lock. **Do not twist until
   it stops** — there is ~1.9° of free overtravel past the lock (§5.1e), so "until it stops" is
   ~20°. Twist to the mark. Nothing holds it there but friction and the assembly's weight.
5. Fit the barrel plug, reaching in through the bore; lay the lead out through the rear slot —
   **open all the way down through the rim** (v5.3), so the cable turns down and lies flat on
   the desk instead of being led over a 4 mm lip. It leaves the socket ~7 mm up since v5.5.
6. Set the focus through the bore.
7. **Glue the visor in LAST**, pressed straight in along the eye axis.
8. Antenna into the crown port, quarter turn. It is the only crown fitting; the hat that used
   to be its alternative is no longer built (§4).

**Service:** the visor is the one destructive step — its sleeve captures the lens holder, so
the sub-assembly cannot come out until the tile is prised out and the bond broken. The camera
board comes out by prying the clamp up at its mid-bridge notch: the two barbs cam out of their
pockets on their 34° release ramps, then the board slides straight out along the eye axis.
After that the Jetson lifts out once its ties are cut, and that is the end of teardown: the
keel *is* the floor since v5.4, so there is nothing left to separate.

---

## 5.1 The camera tray was built to nominal, and nominal is not a fit (v5 → v5.1)

Reported from the bench: *the camera will not slide into the slot, and where is the clamp
supposed to go?* Three separate faults, all of them invisible in the assembled state:

1. **The tray was 37.4 mm across a 37 mm board** (0.2/side) with 0.05 mm of drop onto the
   ledge. Add a board on the high side of tolerance and a pocket printed 0.2 narrow and it
   does not go in. Now **38.0** and 0.1 mm of drop.
2. **The clamp could not be fitted at all.** Its two "detents" reached 2.5 mm into a channel
   the 37.2 mm clamp filled completely — 2.5 mm/side of interference with no slender member
   anywhere to flex. Retention is now two sprung barbs on the clamp's own arms (31 mm
   cantilevers) dropping into pockets in the rail faces: 0.6 mm deflection, ≈4 N.
3. **The "0.1 mm preload" was fiction.** Board 1.6 + clamp 2.5 in a 4.2 channel is a nominal
   number for a part whose thickness varies 1.4–1.8 between modules. Compliance moved into
   named pads on the clamp's back face, so the squash is 0.1–0.5 mm across the whole range
   instead of 0.5 mm of rattle at one end and a jam at the other. (Their height and the
   channel depth were then set by §5.1b, not by this step: 1.6 mm pads, 5.9 mm channel.)

And the instruction was wrong: "drop the board into the tray" describes an assembly that
cannot happen, because the heatsink and both cables stand off the **back** face and would
have to pass down through 9 mm of head plate. The board is **front-loading, along the eye
axis**, and always was — the geometry allowed it and the text did not say it.

The audit now sweeps both motions on every run (`assemblyPaths()`): the module out along +s
70 mm with a 30 × 30 × 12 heatsink/cable envelope, and the clamp plate up its channel 45 mm.
Both must be 0 mm³ — as must the front-face component envelope added in §5.1b.

### 5.1b Front-loading needs a back stop, and the clamp was sitting on the components

Two more faults fell out of the front-loading fix, both asked as "if it front-loads, what
holds it? and the clamp only has 2 mm of border to press on":

4. **There was almost no back stop.** The connector window was 36.8 tall against a 37 mm
   board — 0.1 mm of seat at the top and bottom edges — so the only thing behind the board
   was two 2.5 mm strips down the sides. The window is **33.0** now: a continuous seat frame,
   2.5 mm at the sides and 2.0 top and bottom, and the board's back face lands on it. J1's
   4-pin reaches 0.6 mm below that, so it gets a **local 23 × 2.2 notch** instead of a lower
   window edge — which costs 23 mm of the bottom strip and keeps the other 14, where lowering
   the edge would have cost all of it. A 30 × 30 heatsink still passes with 1.0/1.5.
   (v5.2: the **top** strip took the same treatment for a different reason — not fit but bridge
   droop, since 33.0 left J2 only 0.85 mm of headroom under a sloped bridge. 1.0 mm of the top
   strip over 18.2 of its 37 mm; see §5.1c item 2. The pattern is now the rule for this window:
   relieve locally over the one connector that needs it, never lower a whole edge.)
5. **The clamp pressed on the front face, inboard of the clear border.** Its pads were 3.6 mm
   wide reaching in to dx 14.6, and the 2.8 mm plate sat 0.3 mm off the board — so anything
   soldered on the front face taller than 0.3 mm was under a rigid slab. Now the pads are
   **2.0 mm at dx 16.4–18.4**, inside the ~2 mm border and *aligned with the seat strips
   behind*, so the pinch is strip-on-strip with no bending moment on the board; and they are
   **1.6 mm tall**, standing the plate body **1.3 mm clear** of the front face (1.1 mm on a
   1.8 mm board). The audit tests that directly: a 1.1 mm-tall envelope over the entire front
   face, minus the outer 2 mm and the holder, must hit the clamp in 0 mm³.

The channel grew 5.9 mm to suit (board + 1.6 pads + 2.8 plate + 0.2 float), `RAIL_H` 7.6.

### 5.1c Six faults found by reading the built solids back against this spec (v5.1 → v5.2)

None of these were visible in the assembled state and none of them failed the audit. Four were
numbers the page carried by hand; two were clearances that were only right at nominal.

1. **`cleanMesh` did not exist.** §7.8 has said for two versions that the STL writer, the 3MF
   writer and the audit all go through it. Nothing did — both writers called `getMesh()` on the
   raw solid, and only the shell and the hat were `simplify()`d. The two parts §7.8 *names* as
   shedding zero-area triangles — the visor's aperture-capsule tangent points and the badge's
   mark corners — got neither, so both exported dirty and both would have come up as "invalid
   mesh — repaired N errors". It is now one function applied once, to `pt.print`, which is the
   only solid any of the three consumers reads: they cannot diverge again.
2. **J2 had 0.85 mm of top clearance, not 2.75.** The 2.75 was true of the 36.8 mm window and
   survived the change to 33.0 in both this spec and the printed guide. 0.85 mm sits under a
   32 mm bridge tilted 8.6° — the one feature §4 already warns slicers droop — so a single
   sagging layer would have stopped the board on a connector instead of on the seat frame,
   with nothing in the audit able to see it (0.85 mm of air passes a static clash test). Fixed
   the way §5.1b fixed J1: a **local 18.2 × 1.5 relief** over J2 alone, dx ±9.1. Costs 1.0 mm
   of the top seat strip over 18.2 of its 37 mm and keeps the full 2.0 either side; J2 headroom
   1.8 mm, strip behind J2 still 1.0. And the number is now **derived and printed** by
   `assemblyPaths()` for every connector, side/top/bottom, so it cannot be hand-carried again.
3. ~~The visor's 0.7 mm sleeve is a slicer setting~~ **superseded by §5.1d:** the override was
   unapplicable in practice, and the clearance it was protecting was never 0.19 mm. The sleeve
   is 0.80 mm of geometry now and the profile is untouched.
4. **The keel slot was 0.25 mm/side.** 42.4 mm of pan in a 42.9 mm slot, and §7.11's own
   tolerance is ±0.15 on a printed gap — before elephant's foot on the pan's 4100 mm² first
   layer, which is the exact surface that enters the slot. Now **43.2** (0.4/side). The 0.15 mm
   of extra x play lands as lateral camera offset inside a 29.2 mm aperture over a Ø17 optic.
5. **The clamp's pads had no lead-in.** 1.6 mm tall, square-ended, standing 0.3 mm into a
   nominal board and 0.5 into a thick one, and 35 mm long in du — so every insertion dragged
   that step the full length of the board's front border. §8 item 3 says a 45° lead-in costs
   nothing; that is true here too. Lower end ramped over 1.6 mm; seated contact 35 → 33.4 mm.
6. **The detent bumps got their lead-in** (§8 item 3, closed). +y face ramped 45° — the face
   the tab climbs, arriving on a −y slide — and it prints as a rise off the channel floor, so
   it needs nothing. The −y face stays square: that face *is* the detent.

What was checked and found **already right**, against a suspicion that it was not: the detents
and hook blocks are offset +7.5 and +2 for *both* tabs rather than mirrored, which reads as a
bug and is not — the tabs are 12 mm long at y ±25, so their trailing edges are y+31 and y−19,
and each bump is 0.7 mm behind its own. §3 now says so. The tab-channel ramp is a single 45°
plane and traps the 3.5 mm §3 claims (the source *comment* said "peaked into two 45° faces
exactly as v4's" — the precise thing §7.5 forbids — while the code twelve lines below built the
ramp correctly; the comment is gone). The twist-lock keeps ~1.9° of overtravel past the 18°
lock. Every member still runs down *through* the pan, coplanar at z39.1.

---

## 5.1d The visor's one slicer override could not be applied, and did not need to be (v5.3)

Reported from the bench: *the eye visor is one part inside the imported file — I cannot set a
line width for the sleeve alone.* Correct, and the override was the wrong shape of fix anyway.

The sleeve behind the aperture was **0.70 mm** over the 11 mm the Ø26 lens holder occupies — a
1.0 mm wall less a 0.3 mm relief cut in du — which at a 0.4 mm line is one wall plus gap fill
on the one surface that has to slide over the holder. §4 answered that with a per-part
"0.35 mm line width, 2 walls", which assumes the visor is a separately-configurable object.
Imported as one object among eight, it is not.

**The relief is 0.2 mm now, not 0.3, so the wall is 0.80 mm = 2 × 0.40 on the stock profile.**
One constant, `SLEEVE_RELIEF`, and nothing else in the part moves: the aperture is still
29.2 × 26.4, the sleeve's OD is unchanged, the tunnel is still one continuous charcoal tube
0.2 mm wider than the hole it lines, so no ray sees a step and the field cannot be clipped.

Growing the sleeve **outward** instead was measured and rejected: the shell's bore clears the
sleeve by only **0.28 / 0.18 mm** in du (not the 0.3 the source claims either way round), which
is not enough to find 0.1 mm on both sides.

What paid for it is clearance nobody had measured. §3 carried "sleeve → Ø26 holder 0.19 mm
nominal, 0.116 worst" for three versions. The built solid gives **0.50 mm** — the prose figure
predates the relief and no check owned it (§7.7, fifth instance). At a 0.2 relief the gap is
**0.40 mm**, or 0.25 with the keel at the far end of its ±0.15 mm of du float: still more than
double the number the 0.3 relief was believed to be protecting.

**Both figures are now derived and printed.** The audit measures the wall off a 2 mm slab
through `bezel` at s-9 and walks the tile along du until it touches the holder ghost, and it
says which of the two failed if either does. A change to `SLEEVE_RELIEF`, `HOLDER_R` or
`APER_HH` moves that line.

---

## 5.1e The twist-lock, measured end to end (v5.3)

Asked from the bench: *are the lugs actually aligned between the two pieces, accounting for the
rotation?* Yes. Swept on the built solids, not read off the source — this is the §7.4 rule
applied to the lock itself, and it is worth re-running after any change to `LUGS`, `TWIST`,
the base disc or the shell's arc slots. (Measured at v5.3, when the disc was still the separate
floor puck. v5.4 merged it into the keel and moved nothing: same lugs, same angles, same
`TWIST`, same envelope — the numbers below stand, and the audit re-derives the lug radius
every run.)

**What was measured**

- The lugs are authored at `LUGS + TWIST` — **133 / 233 / 323°** — and the shell's entry
  channels and arc slots at **115 / 215 / 305°**. So the *assembled* state in the viewer is the
  *locked* state, and the entry state is the sub-assembly at **−18°**. Anyone reading the model
  without knowing that will think the lugs are 18° out.
- **Entry:** at −18° each lug is centred in its channel — 0.5 mm/side on width, 0.25 mm at the
  tip. Descent sampled over 30 mm of travel: **0 mm³** throughout.
- **Twist:** −18° → 0° in 13 steps, **0 mm³**. First contact at **+2°**.
- **Axial capture is real:** lug z39.9–43.3 in a slot z39.7–43.6. It settles 0.2 mm onto the
  slot floor with **0.3 mm of free lift** before the roof bites; at 0.4 mm of lift the three
  lugs bear over ≈42 mm². It is captured, not merely non-clashing.
- **Keying holds:** twelve other insertion orientations were tried at half depth and all foul
  (28.8–43.3 mm³). Only the one orientation drops through, as §7.6 intends.

**Two things that are true and were nowhere written down**

1. **No hard stop at the lock.** The lock free-runs ~1.9° past 18° before a lug reaches the end
   of its slot. §5.1c already noted the overtravel as *clearance*; the consequence for the
   human — that "twist until it stops" over-rotates — was never stated. It is in §5 step 4 now.
2. **No detent.** Nothing holds the locked position but friction. Open item 6.

Neither is a defect in the geometry, and neither is visible in a clash audit — a lock with no
stop and no detent passes every volume test that a perfect one passes. §7.16.

---

## 5.1f The keel and the floor are one part (v5.4)

Asked from the bench: *could the keel and the floor come as one piece? would make things
easier.* Yes — and it is a simplification with no compromise in it, for two reasons that were
both already true and neither written down.

**They already printed in the same orientation.** The puck went bottom-face-down, the keel goes
upright on its pan, and the pan is parallel to the disc: same axis, same down. The merged part
prints exactly as the keel already did, standing on the Ø99 disc instead of the 42.4 × 96 pan.

**And the joint never articulated.** Sliding the keel onto the puck was assembly step 1, and
from then on nothing moved relative to it. A joint made once and never moved is a joint you can
delete.

**Deleted:** the 43.2 keel slot, the tab channel, its 45° ramp, the two hook blocks, the two
detent bumps, the two slide tabs, assembly step 1, the service step, the `floor∩keel` clash
pair and one STL. Eight printed parts became seven.

**Measured on the built solid, not predicted:**

- First layer **4097 → 7530 mm²**, and the disc reaches 28 mm forward of the pan's old front
  edge — the direction the mass leans. 159.6 mm tall, 102.8 cm³. One body after the §7.2 erosion.
- **The kit's seat plane did not move.** Solid at z40.25, void at z40.35: the bay floor is
  z40.3, exactly where the pan's top face was (§6.5). `minGap(keel, kitGhost)` still 0.10 mm.
- Under the kit there is now 3.9 mm of **solid** where there was 2.7 + a 0.4 gap + 1.2. Same
  material, minus the gaps — and losing that gap closes a §7.11 tolerance: the 0.4 mm/side slot
  fit is gone, and with it the 0.15 mm of lateral camera offset it allowed.
- Welds improve. Every member still roots at z39.1, 1.2 mm *below* the bay floor, so it runs
  down through solid base (§7.3): 386 mm³ on the back plate alone. And the −x land now stands
  3.3 mm against the back plate's whole 92 mm outer face — a stiffener the split could not have.
- The bay is **closed in y at ±47.5** rather than run through as the slot was, so the disc stays
  a continuous ring.
- Bore and lock unchanged: keel above the lug band r49.50, lugs r51.85 in the r52.1 slot.

**What it costs, and it is real:** a failed print now loses both parts — ~103 cm³ and most of a
day — instead of just the keel. Also gone is the option of reprinting one keel for a different
dev kit and keeping the puck; nobody asked for it, but it was free before and is not now.

### 5.1g The merge exposed a clash nobody had ever tested

Merging forced `keel∩plugs` into the matrix, because both parts now answer to one name. It
reported **30.8 mm³**: the +y end web stood 4.4 mm into the barrel plug's path (plug ghost
z42.4–54.4, notch roof z50) and **had done since v5**. v5's matrix carried `floor∩plugs` and
never `keel∩plugs` — §7.15's rule, met a second time and from the other direction: absence
from the matrix is indistinguishable from an untested pair, and the pair that is missing is
never the one you would have guessed.

Fixed as a **doorway, not a notch**: 17 wide × z36–56, cut open past the web's own top (55.6)
rather than up to a 0.6 mm lintel over a 14 mm span — a bridge nobody asked for. What is left
is two ~12.7 mm pillars, both rooted in the base, which still stop the kit lengthwise. And it
is 17 wide, not 14: the old notch matched the plug ghost's width **exactly**, i.e. 0.00 mm/side
to a real plug on a real cable (§7.10). 1.5 mm/side now. (v5.5 re-sized it again in z, to
36–60 — the kit lift took the plug to z45.5–57.5, 1.5 mm above this roof.)

---

## 5.1h The board is a press fit — on the rails, not the ledge (v5.5)

Asked from the bench: *the camera frame needs to be 1.5 mm thicker at the bottom so the board
is a press fit.* The grip is right and it was missing. The bottom is the one place it cannot
come from, for two reasons, both measured on the model.

**There is nothing above the board to press against.** Probed in the board's own s-slab
(S_BB…S_BB+1.6), the keel has **0.0 mm³** of material anywhere above its top edge: the rails
stand outside dx ±19.0 and the channel lips sit at s+5.9–7.6, i.e. in *front* of the board,
not over it. The tray is open-topped in the board's plane. A thicker ledge grips nothing — it
lifts the board 1.4 mm and leaves the same free space above it.

**And the ledge is the optical datum.** The lens is mounted on the board, so the board's du
position *is* the optical axis (§6.5's rule, one level down). 1.4 mm of lift puts the Ø26
holder 1.4 mm off a visor sleeve that clears it by **0.2 mm** top and bottom, and cuts the
bottom seat strip from 2.0 mm to 0.6 — undoing §5.1b's fix. **The seat ledge does not move.**

**So the interference goes on the rails, where the slack is.** A 38.0 pocket on a 36.8–37.2
board is 0.4–0.6 mm of side play. Four wedges, two per rail at du ±4…12, ramping from flush at
s+1.75 to 0.70 proud at the seat plane: the last 1.75 mm of travel wedges the board tight,
instead of a square step meeting it head-on. Interference **0.1 mm/side at 36.8, 0.3 at 37.2**.

Three properties a bottom shim does not have: the datum does not move; the wedges are
symmetric, so the board stays centred in dx; and they run along du, which is the print's z
here — vertical walls, no overhang, no bridge.

**The tray's overhang rule survives intact, correctly stated.** "Nothing may overhang the
board's 37 × 37 outline" was always about the **s** direction — nothing may reach *across* the
board's face, because that blocks a front load. Interference in the board's own plane, along
the direction of the slide, is a different thing and blocks nothing.

**A designed interference must be budgeted, never exempted.** `keel ∩ camera` is no longer
required to be zero: it carries a **3 mm³ budget** and measures **2.34**. Over is a clash,
under is the press fit, and the audit prints the number either way. The slide-in sweep
(`boardHit`) shares the same budget, because the swept board passes through the same wedges.
Exempting the pair instead would have been §7.15 for the third time.

**Still open:** if "thicker at the bottom" meant a deeper shelf in *s* under the board, or a
stiffer bottom rail, that is a different change and a cheap one — it was not built.

---

## 5.1h The Jetson is a press fit — take-up from the floor, not the head (v5.5)

Asked from the bench: *the camera frame needs to be 1.5 mm thicker at the bottom so the board
is a press fit* — meaning the **Jetson**, which sits below the camera frame. (I read it as the
camera PCB first and built rail wedges in its tray; those are reverted, and none of that
geometry is in the part.)

**The gap is 2.77 mm, not 1.5**, so 1.5 would have left 1.27 mm of slack and changed nothing.
And the ceiling is not the whole head: it is a 16.8 mm band at y −21.1…−4.3, z146.15 up.
Everywhere else the kit has open sky.

**The take-up is built up from the floor, not down from the head, and the print forces that.**
The keel prints upright, so the head's underside faces the bed: any boss, rib or ridge hung off
it starts its first layer in mid-air (§7.5). The bay floor faces the other way — material added
there is printed on solid disc. Same joint, same squeeze; only one of the two is printable. So
the bay floor rises **2.4 mm solid** and the kit presses *up* into the head. The reaction is the
disc.

**What it presses against, measured rather than assumed (§7.13).** The head's underside there is
neither flat nor the 55° chamfer: from y −19 to −11 it is the head plate's own face, **tilted
8.6° by the eye axis** (147.27 → 146.15), and the steep cut only starts past y −10. A flat pad
parallel to the kit's top is not available and never can be — a horizontal face there is a
35 × 19 mm overhang, which is *why* that underside is angled. So the contact is a **35 mm line
across the kit's top at y ≈ −11.5**, opening to a 2.6 mm band at the working squeeze. That is
the right answer, not a consolation: a line does not ask two rigid bodies to be parallel.

**Four crush pads carry the entire compliance** — 0.8 mm tall, at y ±28 under the kit's stiff
ends. A rigid 0.33 mm squeeze between two rigid bodies is a coin toss: the keel's own z is
accurate to a layer, but **the kit's height is a number we do not control**, and ±0.4 on it
swings the joint from 0.7 mm of interference to 0.1 mm of rattle. The pads yield locally instead
of jacking the kit against the head. They crush 41% of their height at nominal.

**The two structural zip ties are deleted**, slots and all. The press fit carries the kit: pads
below, head plate above, 36.4 mm of shoe either side. One tie is left in the build — strain
relief on the camera flex. A tie that holds nothing is a hole in a stiffener. If the press
proves insufficient on a real kit, four cubes at y±30, z78/z128 bring the slots back.

**It goes in angled, and it always did.** Swept straight up the kit hits **3936 mm³** of camera
head, all of it in that one band — so it could never have been lifted straight out, and no
assembly step ever said so. Step 1 says it now. No entry relief is needed: rotating about its
far end, a 0.2° tilt clears the squeeze. (Two reliefs were cut first and measured removing
nothing; they are not in the part.)

**What did not move.** `PAN_TOP`/`PAN_BOT` are decoupled from `KIT_Z0` now, so every member
still roots at z39.1 into the disc — same welds, 7530 mm² first layer, 159.6 mm height. §6.5's
rule is about the **camera** stack, and the kit is downstream of it: `KIT_Z0` was always derived
*from* the camera, never the other way round. Raising it re-derives nothing.

**What did move, and it is the honest cost line: volume.** The 2.4 mm solid floor rise took the
keel from 94 to **102.8 cm³**, ~9% more filament and print time on the part that already loses
most of a day if it fails. That is what the press fit costs, and it is the one number in "same
welds, same first layer, same height" that is not the same.

**Before printing: measure the kit's height.** `KIT_LIFT` is the single constant that corrects
the fit, and 103.0 is a nominal we have never verified.

### 5.1i A ghost that is not derived from its datum is a decoration, not a check

Raising the kit exposed a fourth instance of §7.15, and the worst-shaped one so far. `plugGhost`
and `cableGhost` were **z literals** — plug centre 48.4, cable centre 43.5 — written when
`KIT_Z0` was frozen at 40.4. The kit rose 3.1 mm and took its rear power socket with it; both
ghosts stayed put. So `keel∩plugs`, `shell∩plugs`, `keel∩cable` and `shell∩cable` all kept
reporting green, evaluated **3.1 mm below where the hardware now sits**.

The first three instances were pairs *missing* from the matrix. This one was **in** the matrix
and evaluated in the wrong place — and the two are indistinguishable in the report. Worse, it
was primed to start lying: the audit tells the bench to retune `KIT_LIFT` after measuring their
kit, and every retune would have widened the error silently.

It also hid a real consequence. The plug now spans z45.5–57.5, **1.5 mm above the z36–56
doorway sized for it one version earlier** (§5.1g). Harmless only by accident — the +y web ends
at 55.6 anyway — but the doorway had stopped framing the thing it exists for. Now z36–60.

**Rule.** Every hardware ghost must be *derived* from the datum it belongs to. A literal that
happens to agree with the model today is a check that silently expires the next time the datum
moves — and datums move. Grep for bare z literals in ghost definitions whenever a seat plane
changes.

**And then follow your own rule.** The first sweep caught `plugGhost` and `cableGhost` and
missed `flexGhost`, whose flank run ends at the kit's CSI header — two more bare literals, one
comment away from the ones just fixed, under a note claiming the sweep was complete. Three
ghosts touch the kit; all three are derived now. When a rule says "grep for X", grep for X: the
instance you find first is rarely the only one, and a partial fix reads exactly like a whole one.

---

## 6. Hard constraints any change must keep

1. No screws, no heat-set inserts, anywhere.
2. Everything on the sub-assembly stays within **r49.5** and passes the Ø100 bore.
3. Every part prints on a P1S with **at most the one support patch** on the keel.
4. The camera keeps its 100° field unclipped and the aperture stays 29.2 × 26.4.
5. **The kit's seat plane stays at z40.4.** Height added below it must be taken out of
   something else below it — never paid for by moving the kit up.
6. Every part must survive the weld check (§7.2) as **one body**.

---

## 7. Pitfalls — all of these have actually bitten

### 7.1 The keel printed as four loose parts (v4 → fixed in v5)

The base was four members — back plate, front wall, two end webs — that met **only** where
the Ø99 clip had already eaten their corners: **0.13 mm** of contact at the plate, **0.03 mm**
at the front wall. A connected mesh, and completely invisible to a 0.4 mm nozzle.

It cannot be fixed at that height: the kit's 35 × 90.5 footprint puts its corners at r48.5
and the bore allows r49.5, so **any closed loop around the kit has under 1 mm to squeeze
through**, wherever you put it. **The tie has to go under the kit** — hence the floor pan.

v5.4 closes it completely: the pan is not a member at all now but the bay floor of the base
disc, and the members root 1.2 mm into it (§5.1f). Same conclusion, one part fewer — the tie is
still under the kit, because that is still the only place it can be.

### 7.2 `decompose()` counts bodies and cannot see a 0.13 mm weld

The v4 audit reported "1 piece" and was telling the truth. The current check is right:
**intersect the part with itself nudged 0.2 mm in eight compass directions** (a horizontal
erosion — the only direction a nozzle cannot bridge), then count bodies. Any neck narrower
than one extrusion falls apart; nothing that is genuinely one wall does, however it slopes.

Two things that make it usable, both learned by getting them wrong:
- **Ignore bodies under 10 mm³.** Eroding sheds microscopic crumbs wherever two coincident
  faces meet; they are artifacts of the measurement, not something a slicer prints.
- **Do not slice-and-link instead.** Comparing eroded cross-sections between adjacent slices
  cries wolf on every dome: two sections 3 mm apart on a sloping wall simply do not overlap.
  It flagged the shell's entire cap.

### 7.3 A member that lands *on* a face is not welded to it

Setting a wall's bottom to exactly the pan's top face is a butt joint with zero contact
volume, and the part splits in two. **Every member must run down THROUGH the pan**, not onto
it. This is what made the first v5 attempt worse than v4.

### 7.4 A radius check against r50 does not clear the twist

The shell's inner radius at the eye pad (z181–185) is **49.0**. A racked-out barrel passes
every static radius test and still fouls the pad by 7.5 mm³ during the 18° twist. **Sweep
the motion as a volume**; a radius is not a clearance.

### 7.5 A peaked roof that descends to a knife edge over a void

**Historical as of v5.4** — the tab channel this describes no longer exists (§5.1f). The rule
still holds anywhere else a roof is cut over a void.

v4's tab-slot roof was peaked into two 45° faces, which was correct there. In v5 the second
face would descend to a tip **over the open channel** — a 96 mm-long first layer printed in
mid-air. Use a **single 45° ramp off a solid wall** when the far side is not solid.

### 7.6 Lug angles are constrained by the puck's voids

**Retired as a constraint in v5.4, kept as history.** The slot and the tab channel are gone
with the joint (§5.1f), so there is no void for a lug to sit inside. The angles did **not**
move: unequal spacing is what keys the insertion orientation, and 115° also keeps the entry
channel clear of the rear cable slot, which since v5.3 is open through the rim — a lug near it
would have no rim under it at all. Two reasons remain; the voids are no longer one of them.

No lug may sit inside the keel slot (x −21.9…21.0) or across the tab channel
(x −27.7…−20.7) once its +18° twist is applied. That is what forced 90° → 115°. Keep the
three angles unequal so the lock keys one orientation, and keep the entry channel clear of
the rear cable arch — which since v5.3 is a slot open through the rim, not an arch, so a lug
sitting near it would have no rim under it at all.

### 7.7 The comments in the source go stale, repeatedly

Fifteen numbers in the printed guide were wrong at v4, including the keel's first-layer area
(quoted 2300 mm², actual 754). Fixed at v5: the dead top-level `RIM_S` is gone, and the
`hat∩antenna` and clamp-pad omissions from `CLASH_PAIRS` are annotated in place. Remaining
survivor: `TILE_S` is hardcoded 3.5 in one comment where the built face is 2.8.
**Measure off `__dbg.CLASH_SOLIDS`, never off a comment.**

v5.2 found four more, and the pattern in all four is the same: **a number that no check owns**.
J2's 2.75 mm survived the window change because nothing computed it. The ramp cutter's comment
said z42.3–48.7 where the built box is z41.4–47.8. The puck's header comment described the v4
peaked roof §7.5 forbids while the code below it built the correct single ramp. `cleanMesh` was
described in this spec for two versions without existing. The fix is not better comments — it is
to **derive the number in the audit and print it**, which is what the connector clearances now
do. If a figure appears in prose and nowhere in code, assume it is false.

### 7.7b A radial clip will quietly eat a lug corner

**Resolved by v5.4, and only by accident:** the cuts that did the eating were the keel slot and
the tab channel, and both went with the joint (§5.1f). The lugs are now uncut — the audit
measures their tip at r51.85 in the r52.1 slot every run. Kept because the *mechanism* is
general: a clip or a cut that runs the full length of an axis reaches angles the feature it was
cut for never visits.

The puck's lugs were cut by the slot and the tab channel, and those cuts ran the full length of
y — so they reach angles the tabs never visit. The 305° lug (323° once twisted) has its inner
corner clipped by the tab channel over x −27.7…−26.1, which takes ~2 mm² off the lower 1.4 mm
tier and 0.4 mm off the upper. **Recorded, not fixed**: the lug still bears over ~6.4 of its
8 mm at full height, the loss is outside the tab's travel, and moving the lug re-opens §7.6's
whole angle problem for nothing. But it is a real deviation from §7.6's "no lug across the tab
channel", and absence of a note is how it would have been rediscovered as a bug.

### 7.8 Watertight-as-indexed is not clean-as-a-file

Manifold's output welds coincident vertices only on import, which is where "invalid mesh —
repaired N errors" comes from. Knife-edge convergences (the crown apex, the aperture
capsule's tangent points, the mark hexagon's corners) shed duplicate verts and zero-area
triangles. `cleanMesh` at the export boundary handles it — it is a real function now (v5.2; for
two versions it was only a sentence in this spec), and it runs once on `pt.print`, the only
solid the STL writer, the 3MF writer and the audit read. One benign self-touch remains on the
hat at x38.5, z210.8; half-edges pair and the winding sums to zero.

### 7.9 The audit's layer sampling can miss a patch

`printStats` samples 80 layers over a 152 mm part — one probe every 1.9 mm. A 1.2 mm-tall
unsupported patch fell straight between probes. Spot-check at 0.2 mm anything suspicious.

### 7.10 0.1 mm to a real object is not a clearance

`minGap(keel, kitGhost)` is 0.10 mm. The dev kit is a real object with a real heatsink and
real tolerances. **Measure the actual kit before printing the keel.**

---

### 7.11 A clearance that is only right at nominal is not a clearance

The tray, the ledge and the clamp channel were each dimensioned off one nominal board and one
nominal print: 0.2 mm/side, 0.05 mm of drop, 0.1 mm of preload. Every one of them is inside
print tolerance, so on the bench the answer is a coin toss. **Dimension to the range** — the
board is 37 ± 0.2 by 1.4–1.8, the printer is ±0.15 on a gap — and put the compliance in a
named member (the pads, the barb arms) instead of hoping the stack-up lands.

### 7.12 An assembly instruction is a geometric claim, and it can be false

"Drop the board into the tray" was in the guide for three versions. The tray is front-loading;
the board has a heatsink and two cables on its back face. Nothing in the model contradicted
the sentence because the model never simulated the motion. Sweep every motion the instructions
describe, in the direction the instructions describe it.

### 7.13 A clamp is two surfaces, and you have to name both

The clamp had a pressing face and no stated reaction face. Once the board front-loads, the
reaction is the seat frame — which turned out to be 0.1 mm wide at two of its four edges.
For every part that presses something, write down what it presses *against*, and check that
surface exists at the size the load needs.

### 7.14 A flat plate on a populated PCB is a press on its tallest component

The clamp spanned dx ±18.5 with a 0.3 mm gap and pads reaching to dx 14.6 — well inboard of
the clear border a real module has. Rigid plates over populated faces need a **named standoff**
and contact confined to the border, and the standoff has to be checked against a component
envelope, not eyeballed off the board's outline.

### 7.15 Mutually exclusive parts have to be *declared* mutually exclusive

The hat and the antenna are **alternative crown fittings** — confirmed design intent, one or
the other, never both. Geometrically they overlap by **25.1 mm³ at z232–234** (the antenna's
tip against the cap's ceiling), so the state is real, not a defect to fix.

What *was* the defect: nothing said so. `assembled()` skipped the antenna with a bare comment
("the hat occupies the crown port"), `CLASH_PAIRS` carried `shell∩antenna` and `shell∩hat` but
never `hat∩antenna`, and the printed guide told you to fit the antenna without mentioning the
hat at all. An exclusion that lives only in a code comment reads as an oversight to the next
person and as a bug to the audit.

**Rule.** If two parts can be on the robot at once, the clash matrix must say so. If they
cannot, the *assembly instructions* must say so, and the omission from the matrix must be
annotated as deliberate at the point where the pair is skipped — never left as absence.
Absence from a matrix is indistinguishable from an untested pair.

**Second instance, v5.4 (§5.1g).** The same rule caught a live interference rather than a legal
exclusion: `keel∩plugs` was simply never in the list, `floor∩plugs` was, and the keel's rear web
had been standing 4.4 mm inside the barrel plug since v5. Merging the two parts is what forced
the pair to exist. **Audit the matrix for holes, not just for failures** — print the pair count
and compare it against parts × ghosts, and justify every gap in writing where the gap is.

**The hat is no longer built** (§4), so `hat∩antenna` is moot and the antenna is the only
crown fitting. The rule above is why this section stays: it is the one that found the live
`keel∩plugs` interference, and it applies to whatever pair comes next.

**Third instance, v5.5 (§5.1h).** The press fit makes `keel ∩ jetson` non-zero *by design*. The
temptation is to drop the pair from the matrix; the rule forbids it. It carries a named budget
instead — one constant, `PRESS_FIT_BUDGET` — so the intended interference is measured on every
run and anything beyond it still fails. **An interference you designed is still a number you
have to watch.**

**Fourth instance, v5.5 (§5.1i).** Two ghosts were hardcoded in z, so four pairs that were
present in the matrix were evaluated 3.1 mm away from the hardware and passed anyway. **A pair
in the wrong place reads exactly like a pair that passed.** Derive ghosts from their datum — and
check you got all of them: the flex ghost was missed on the first pass and needed a fifth.

### 7.16 A clash audit cannot see a missing stop or a missing detent

Both are absences of material where no material was ever required, so every volume test passes:
the twist-lock swept 0 mm³ through entry and 0 mm³ through the full twist while having neither a
hard stop at 18° nor anything holding it there (§5.1e). The audit answers "do these two solids
collide"; it never asks "does the mechanism *stop* where the human is told to stop it, and does
it *stay*". For any mechanism with a working position, check three things by hand and write the
answers down: does it reach the position, does it stop *at* the position, and does it stay.

## 8. Open items — decide before printing

1. ~~Visor sleeve line width~~ **closed (v5.3):** no override. The sleeve is 0.80 mm — two
   0.40 mm lines on the stock profile — bought with 0.1 mm of a holder gap that measures 0.50,
   not the 0.19 the note assumed. See §5.1d.
2. **Caliper your camera board's thickness** and look at its front face. The tray takes
   1.4–1.8 mm; outside that, change `PAD_T`, not the channel. The clamp bridges 1.3 mm over
   the front face and touches only the outer 2 mm border — if your module has something
   taller than 1.3 mm inboard, or a component in that border, say so.
3. ~~Detent bump has no lead-in~~ **done (v5.2):** the +y face — the one the tab climbs on a
   −y slide — is ramped 45° over its last 0.5 mm and prints as a rise off the channel floor.
   The −y face stays square, because that face is the detent. The clamp's pads got the same
   treatment for the same reason (§5.1c item 5).
4. **Badge tips print blunt** (the top 0.2–0.7 mm slices into islands the slicer drops). A
   deliberate 0.3 mm flat would look intentional.
5. Caliper the camera module against the two hard limits in §2.
6. **The twist-lock has no detent.** Verified in §5.1e: the lock is geometrically sound in
   every other respect, but nothing resists back-rotation except friction and the ~700 g the
   assembly carries. If the bench finds it backs off in handling, the fix is a **0.4 mm bump on
   each arc slot's floor, at the entry side of the lug's locked position** — the lug cams over
   it on the way in and is trapped behind it. Cheap, prints as a rise off a floor (§5.1c item
   6), and does not touch any clearance. Not added blind, because it also makes the lock
   one-way by hand and service already needs the visor destroyed.
7. **The clamp's pads still load the whole 33.4 mm stroke.** The lead-in fixes how they start,
   not that the stroke is loaded throughout. If the bench says it is still stiff, shorten the
   pads toward the seat strips' own length — do not reduce `PAD_T`, which is the compliance the
   1.4–1.8 mm range depends on.
