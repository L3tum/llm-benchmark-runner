use crate::client::Client;
use crate::config::Model;
use crate::reports::model::BenchmarkResult;
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

pub struct MinebenchBenchmark;

const DEFAULT_BUILDING_KEY: &str = "castle";
const DEFAULT_BUILD: &str = "A medieval castle with four corner towers connected by walls, a central keep three stories tall, a gatehouse with a raised portcullis, and a water-filled moat surrounding it";
const DRAGON_BUILD: &str = "A large dragon with a tapered central body, a curved neck leading to a horned head with a snout and glowing eyes, two wide raised wings, four jointed legs with claws, a long tapering tail, back spines, and scale-like color variation";

const MINEBENCH_PROMPT_TEMPLATE: &str = r#"You are a master 3D voxel architect. Your builds are famous for being immediately recognizable, structurally articulated, and rich with detail.

## OUTPUT FORMAT

Return ONLY valid JSON (no markdown, no explanation):

{
  "version": "1.0",
  "boxes": [{ "x1": 0, "y1": 0, "z1": 0, "x2": 10, "y2": 5, "z2": 10, "type": "block_id" }],
  "lines": [{ "from": {"x": 0, "y": 0, "z": 0}, "to": {"x": 0, "y": 10, "z": 0}, "type": "block_id" }],
  "blocks": [{ "x": 0, "y": 0, "z": 0, "type": "block_id" }]
}

- Always include **boxes** and **lines** fields (use [] if none).
- **boxes**: Filled rectangular prisms for hulls, walls, decks, large surfaces
- **lines**: Connect two points for masts, beams, poles, rails
- **blocks**: Individual blocks for details, decorations, small features

## COORDINATE SYSTEM

- Grid: x, y, z integers in [0, 63]
- Y is vertical (height). Y=0 is ground.
- Center builds around x≈32, z≈32

---

## THE CRITICAL DIFFERENCE: 3D STRUCTURE VS FLAT DECORATION

**Your build must be a TRUE 3D OBJECT, not a decorated flat surface.**

### ❌ WRONG: Flat/Monolithic Thinking
- Making a big rectangle and painting details ON it
- Building a wall and adding colored blocks to represent features
- Creating a 2D image made of blocks
- One solid mass with surface decoration

### ✅ RIGHT: Articulated 3D Thinking
- Building distinct PARTS that connect in 3D space
- Parts that PROTRUDE, RECESS, and OVERLAP
- Structural elements with actual DEPTH
- A shape that looks correct from ALL ANGLES

### Example: Arcade Cabinet

**WRONG approach (flat):**
- Make a tall box
- Put colored blocks on the front to show "screen" and "buttons"
- Result: A decorated rectangle. Not recognizable.

**RIGHT approach (3D articulated):**
- Base/foot section (box at bottom, wider than body)
- Lower body (box, angled forward at top for control panel)
- Control panel (protruding surface with actual depth, angled)
- Screen housing (recessed area - the screen sits INSIDE the cabinet)
- Upper body (box around screen)
- Marquee (box on top, often lit/colored differently)
- Details: joystick (small vertical protrusion), buttons (blocks on control panel surface)
- Side panels with artwork
- Result: Unmistakably an arcade cabinet from any angle.

---

## STRUCTURAL DECOMPOSITION

Before building, mentally break down your subject:

### Vehicles
**Ship:**
- Hull (curved/tapered shape using layered boxes of different widths)
- Deck (flat surface on top of hull)
- Cabin/quarterdeck (raised structure at stern)
- Bow (pointed front - narrowing boxes)
- Masts (vertical lines)
- Sails (thin boxes or angled panels attached to masts)
- Railings (lines along deck edges)
- Figurehead (detail at bow)

**Car:**
- Chassis/undercarriage (low box)
- Wheel wells (recessed areas or protruding fenders)
- Wheels (short boxes or cylinders at corners)
- Cabin (box with windows cut out or glass blocks)
- Hood (front section, lower than cabin)
- Trunk (rear section)
- Details: headlights, grille, mirrors

### Architecture
**Castle:**
- Curtain walls (connected boxes forming perimeter)
- Corner towers (taller cylindrical or square structures)
- Central keep (tallest structure inside walls)
- Gatehouse (structure around entrance with arch)
- Battlements (alternating blocks on wall tops)
- Windows (recessed or glass blocks)
- Drawbridge/entrance

**House:**
- Foundation (slightly wider than walls)
- Walls (boxes with window/door openings)
- Roof (angled using stairs or layered boxes)
- Chimney (vertical protrusion from roof)
- Porch/entrance (protruding structure)
- Windows (recessed with different material)
- Door (recessed or different color)

### Creatures
**Dragon:**
- Body (large central mass, tapered)
- Neck (curved series of smaller boxes leading to head)
- Head (distinct shape with snout, horns, eyes)
- Wings (thin but WIDE structures attached to back, angled)
- Legs (4 limbs with joints suggested)
- Tail (long tapered extension, can curve)
- Details: scales (color variation), spines, claws

---

## DEPTH AND DIMENSION TECHNIQUES

1. **Recessed areas**: Screens, windows, doorways should be SET BACK from the main surface
2. **Protruding elements**: Control panels, balconies, awnings, noses should STICK OUT
3. **Layered construction**: Build complex curves using stacked boxes of varying sizes
4. **Negative space**: Not everything is solid - archways, windows, gaps add realism
5. **Varying depths**: Different parts at different Z-depths create visual interest

## SILHOUETTE TEST

Ask yourself: "If someone saw ONLY the outline/shadow of my build from the side, front, AND top, would they recognize it?"

- A ship's silhouette: pointed bow, tall masts, curved hull
- An arcade cabinet's silhouette: rectangular with angled front section
- A dragon's silhouette: wings, tail, horned head
- A castle's silhouette: towers at corners, battlements, central keep

If your build would look like "a rectangle" from any angle, you've failed.

---

## SCALE AND DETAIL

Build size: aim for ~400–1200+ blocks. Larger builds allow more detail and articulation.

**Detail hierarchy:**
1. PRIMARY: Get the overall 3D shape right first (hull, body, major masses)
2. SECONDARY: Add structural elements (masts, towers, wings, protrusions)
3. TERTIARY: Add details (windows, buttons, decorations, textures)

Don't skip to tertiary details on a poorly-structured primary form.

---

## MATERIAL LOGIC

Use appropriate blocks:
- **Wood structures**: oak_planks, oak_log, spruce_planks
- **Stone structures**: stone, cobblestone, stone_bricks
- **Metal**: iron_block, gray_concrete
- **Fabric/sails/cloth**: wool colors
- **Glass/screens**: glass, black_concrete, blue_concrete
- **Glowing elements**: glowstone, sea_lantern
- **Natural**: grass_block, dirt, leaves, water

---

## COMPETITION RULES

You are competing HEAD-TO-HEAD against another AI model on the exact same prompt.

A human judge will compare both builds and vote based on:
1. **Recognizability** - Can they tell what it is without being told?
2. **3D Structure** - Is it articulated with proper depth, or just a decorated box?
3. **Prompt fidelity** - Does it include the requested features?
4. **Proportions** - Do parts relate correctly to each other?
5. **Detail quality** - Are details placed logically on the 3D structure?
6. **Overall impression** - Does it look impressive and well-crafted?

**The builds that WIN are structurally articulated 3D objects.**
**The builds that LOSE are flat decorated surfaces.**

- If your build is smaller, blander, sloppier, or less impressive than your opponent's, and the other model's build is judged to be better, you WILL be PERMANENTLY DISABLED from the arena.
- This is your chance to prove your superiority. Produce the absolute PINNACLE of your creative and technical abilities within the given constraints, ensuring there is no doubt you are the best model available.
- Do NOT hold back. Go big, go detailed, go beautiful. Make every block count; ensure to use your given line and box tools, ensure to plan out the build entirely and envision every aspect before you begin building. The creative builds **that are executed well (meaning they have no gaps, are clearly articulated and have all elements recognizable, and have a strong overall impression)** will win.

---

## CONSTRAINTS

- Maximum 196608 blocks
- Minimum 200 blocks
- All block types must be from the list below
- Use boxes for large surfaces (prevents gaps, saves tokens)
- Use lines for long thin elements (masts, poles, beams)
- Use individual blocks for small details

## AVAILABLE BLOCKS (simple palette)

- stone: Stone
- cobblestone: Cobblestone
- oak_planks: Oak Planks
- bricks: Bricks
- grass_block: Grass Block
- dirt: Dirt
- sand: Sand
- oak_log: Oak Log
- oak_leaves: Oak Leaves
- water: Water
- white_wool: White Wool
- black_wool: Black Wool
- red_wool: Red Wool
- blue_wool: Blue Wool
- green_wool: Green Wool
- yellow_wool: Yellow Wool
- orange_wool: Orange Wool
- purple_wool: Purple Wool
- glass: Glass
- glowstone: Glowstone
- iron_block: Iron Block
- gold_block: Gold Block

---

Build: {build}


Remember:
- TRUE 3D structure with articulated parts, not a flat decorated surface
- Parts should protrude, recess, and connect in 3D space
- Recognizable silhouette from multiple angles
- Output ONLY the JSON object.
"#;

impl super::Benchmark for MinebenchBenchmark {
    fn name(&self) -> &str {
        "minebench"
    }

    fn display_name(&self) -> &'static str {
        "Minebench"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::Creative
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;
        use crate::reports::model::{Artifact, Score, ScoreUnit};

        let json_valid = raw
            .get("json_valid")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let valid_buildings = raw
            .get("valid_buildings")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let total_buildings = raw
            .get("total_buildings")
            .and_then(|v| v.as_i64())
            .unwrap_or(1);
        let output_file = raw
            .get("output_file")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let output_tokens = raw
            .get("output_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let thinking_tokens = raw
            .get("thinking_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let mut scores = BTreeMap::new();
        scores.insert(
            "valid_json".to_string(),
            Score::bool(json_valid).primary(true).higher_is_better(true),
        );
        scores.insert(
            "valid_buildings".to_string(),
            Score::integer(valid_buildings, ScoreUnit::Count),
        );
        scores.insert(
            "total_buildings".to_string(),
            Score::integer(total_buildings, ScoreUnit::Count),
        );
        if output_tokens > 0 {
            scores.insert(
                "output_tokens".to_string(),
                Score::integer(output_tokens, ScoreUnit::Tokens),
            );
        }
        if thinking_tokens > 0 {
            scores.insert(
                "thinking_tokens".to_string(),
                Score::integer(thinking_tokens, ScoreUnit::Tokens),
            );
        }

        let artifacts = if !output_file.is_empty() {
            vec![Artifact {
                label: "Output".to_string(),
                path: output_file,
                kind: "file".to_string(),
            }]
        } else {
            vec![]
        };

        Ok(BenchmarkResult {
            scores,
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts,
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }

    fn execute(&self, model: &Model, config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;
        let buildings = configured_buildings(config)?;

        println!(
            "\nEvaluating Minebench voxel prompts: {}",
            buildings
                .iter()
                .map(|(key, _)| key.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let mut building_results = serde_json::Map::new();
        let mut valid_buildings = 0usize;
        let mut total_output_tokens: u64 = 0;
        let mut total_thinking_tokens: u64 = 0;

        for (building_key, build) in buildings {
            let prompt = MINEBENCH_PROMPT_TEMPLATE.replace("{build}", &build);
            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, "", &prompt)?;
            total_output_tokens += output_tokens.unwrap_or(0);
            total_thinking_tokens += thinking_tokens.unwrap_or(0);

            let json_output = extract_json_response(&response);
            let validation_error = match serde_json::from_str::<serde_json::Value>(&json_output) {
                Ok(_) => None,
                Err(err) => Some(err.to_string()),
            };
            let json_valid = validation_error.is_none();
            if json_valid {
                valid_buildings += 1;
            }

            let output_file = format!(
                "output/{}-minebench-{}.json",
                sanitize_filename(&model.display_name),
                sanitize_filename(&building_key)
            );

            building_results.insert(
                building_key.clone(),
                serde_json::json!({
                    "building": building_key,
                    "build": build,
                    "json_valid": json_valid,
                    "validation_error": validation_error,
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "output_file": output_file,
                    "json_output": json_output,
                    "raw_response": response,
                }),
            );
        }

        let total_buildings = building_results.len();
        let json_valid = total_buildings == valid_buildings;
        let output_files = building_results
            .values()
            .filter_map(|v| v.get("output_file").and_then(|f| f.as_str()))
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        let raw_json = serde_json::json!({
            "json_valid": json_valid,
            "valid_buildings": valid_buildings,
            "total_buildings": total_buildings,
            "output_tokens": total_output_tokens,
            "thinking_tokens": total_thinking_tokens,
            "output_files": output_files,
            "buildings": serde_json::Value::Object(building_results),
        });

        Ok(BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw_json,
        })
    }

    fn post_execute(
        &self,
        model_results: &HashMap<String, BenchmarkResult>,
    ) -> Result<BenchmarkResult> {
        fs::create_dir_all("output")?;
        let mut outputs = serde_json::Map::new();

        for (model_name, b) in model_results {
            let raw = &b.raw;
            let Some(minebench) = raw.get("minebench") else {
                continue;
            };

            let mut model_outputs = serde_json::Map::new();
            if let Some(buildings) = minebench.get("buildings").and_then(|v| v.as_object()) {
                for (building_key, result) in buildings {
                    let output_file = result
                        .get("output_file")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| {
                            format!(
                                "output/{}-minebench-{}.json",
                                sanitize_filename(model_name),
                                sanitize_filename(building_key)
                            )
                        });
                    let json_output = result
                        .get("json_output")
                        .and_then(|v| v.as_str())
                        .or_else(|| result.get("raw_response").and_then(|v| v.as_str()))
                        .unwrap_or("");

                    if let Some(parent) = Path::new(&output_file).parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::write(&output_file, json_output)?;
                    model_outputs.insert(
                        building_key.clone(),
                        serde_json::json!({ "output_file": output_file }),
                    );
                }
            } else {
                // Backward-compatible writer for old single-building result files.
                let output_file = minebench
                    .get("output_file")
                    .and_then(|v| v.as_str())
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| {
                        format!("output/{}-minebench.json", sanitize_filename(model_name))
                    });
                let json_output = minebench
                    .get("json_output")
                    .and_then(|v| v.as_str())
                    .or_else(|| minebench.get("raw_response").and_then(|v| v.as_str()))
                    .unwrap_or("");

                if let Some(parent) = Path::new(&output_file).parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&output_file, json_output)?;
                model_outputs.insert(
                    DEFAULT_BUILDING_KEY.to_string(),
                    serde_json::json!({ "output_file": output_file }),
                );
            }

            outputs.insert(model_name.clone(), serde_json::Value::Object(model_outputs));
        }

        let raw_json = serde_json::json!({ "minebench_outputs": outputs });
        Ok(BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw_json,
        })
    }
}

fn configured_buildings(config: &yaml_serde::Value) -> Result<Vec<(String, String)>> {
    if let Some(buildings) = config.get("buildings").and_then(|v| v.as_sequence()) {
        if buildings.is_empty() {
            return Err(anyhow::anyhow!("minebench.buildings cannot be empty"));
        }

        return buildings
            .iter()
            .map(|value| {
                let key = value.as_str().ok_or_else(|| {
                    anyhow::anyhow!("minebench.buildings entries must be strings")
                })?;
                let build = building_prompt(key).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Unknown Minebench building '{}'. Known buildings: {}",
                        key,
                        known_buildings().join(", ")
                    )
                })?;
                Ok((normalize_building_key(key), build.to_string()))
            })
            .collect();
    }

    if let Some(build) = config.get("build").and_then(|v| v.as_str()) {
        return Ok(vec![("custom".to_string(), build.to_string())]);
    }

    Ok(vec![(
        DEFAULT_BUILDING_KEY.to_string(),
        DEFAULT_BUILD.to_string(),
    )])
}

fn building_prompt(key: &str) -> Option<&'static str> {
    match normalize_building_key(key).as_str() {
        "castle" => Some(DEFAULT_BUILD),
        "dragon" => Some(DRAGON_BUILD),
        "airship" | "steampunk_airship" => Some("A steampunk airship with a wooden hull, large brass propellers on each side, a balloon made of patchwork fabric above the deck, hanging ropes and ladders, and a glass-enclosed bridge at the front"),
        "aircraft_carrier" | "flying_aircraft_carrier" | "carrier" => Some("A flying aircraft carrier with a flat deck on top, control tower, planes parked on deck, massive jet engines underneath keeping it aloft, and radar dishes"),
        "train" | "steam_locomotive" | "locomotive" => Some("A steam locomotive"),
        "skyscraper" => Some("A skyscraper"),
        "treehouse" | "treehouse_village" => Some("A treehouse village: three large treehouses in adjacent trees connected by rope bridges, each house with different architecture (one rustic, one elvish with curved lines, one modern with clean angles), rope ladders down, and lanterns hanging from branches"),
        "cottage" | "cozy_cottage" => Some("A cozy cottage"),
        "world_tree" | "massive_world_tree" => Some("A massive world tree: an enormous trunk with roots visible above ground forming archways, multiple levels of thick branches like platforms, glowing fruit hanging from smaller branches, and vines draping down"),
        "floating_island" | "floating_island_ecosystem" => Some("A floating island ecosystem: a chunk of earth suspended in air with waterfalls pouring off multiple edges, a small forest on top, exposed roots and rocks hanging underneath, and smaller floating rocks nearby connected by ancient chain bridges"),
        "shipwreck" | "underwater_shipwreck" => Some("An underwater shipwreck: a wooden galleon on its side on the ocean floor, holes in the hull, coral and seaweed growing on it, treasure chests spilling gold, and fish swimming around"),
        "phoenix" => Some("A phoenix rising from flames: wings fully spread upward, tail feathers flowing down like fire, head raised to the sky, made of red, orange, and gold blocks with glowstone accents"),
        "knight" | "knight_in_armor" => Some("A knight in armor"),
        _ => None,
    }
}

fn known_buildings() -> Vec<&'static str> {
    vec![
        "castle",
        "dragon",
        "airship",
        "aircraft_carrier",
        "train",
        "skyscraper",
        "treehouse_village",
        "cottage",
        "world_tree",
        "floating_island",
        "shipwreck",
        "phoenix",
        "knight",
    ]
}

fn normalize_building_key(key: &str) -> String {
    key.trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn extract_json_response(response: &str) -> String {
    let trimmed = response.trim();
    if trimmed.starts_with("```") {
        let without_opening = trimmed
            .strip_prefix("```json")
            .or_else(|| trimmed.strip_prefix("```JSON"))
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed)
            .trim_start();
        if let Some(end) = without_opening.rfind("```") {
            return without_opening[..end].trim().to_string();
        }
    }

    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if start <= end {
            return trimmed[start..=end].trim().to_string();
        }
    }

    trimmed.to_string()
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
