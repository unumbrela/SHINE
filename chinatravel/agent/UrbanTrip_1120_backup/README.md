# UrbanTrip Agent - Documentation

## 📁 Module Structure

```
chinatravel/agent/UrbanTrip/
├── urbantrip_agent.py          (3216 lines) - Core agent with DFS search
├── constraint_parser.py        - Parse user constraints from DSL
├── constraint_validator.py     - Validate plans against constraints
├── ranking_engine.py           - Rank POIs and transports
├── poi_manager.py              - Manage POI data collection
├── plan_builder.py             - Construct travel plans
├── search_engine.py            - Alternative search implementation (not used)
└── urbantrip_utils.py          - Utility functions
```

## 📖 urbantrip_agent.py Structure

### File Organization (3216 lines total)

| Section | Lines | Description |
|---------|-------|-------------|
| **Header** | 1-59 | Imports and module setup |
| **Class Doc** | 60-85 | Architecture overview and file map |
| **Section 1** | 86-113 | Initialization (`__init__`) |
| **Section 2** | 114-146 | NL2SL Translation (`translate_nl2sl`) |
| **Section 3** | 147-303 | Main Execution Flow (`run`, `symbolic_search`) |
| **Section 4** | 304-883 | Search Logic - `generate_plan_with_search` (563 lines) |
| **Section 5** | 884-2671 | Search Logic - `dfs_poi` (1765 lines) |
| **Section 6** | 2672-2872 | Constraint Validation (200 lines) |
| **Section 7** | 2873-2933 | Helper Methods (60 lines) |
| **Section 8** | 3130-3213 | Delegation Methods (83 lines) |

### Quick Navigation

To jump to a specific section in your editor:

```bash
# Section 1: Initialization
:86

# Section 2: NL2SL Translation
:114

# Section 3: Main Flow
:147

# Section 4: generate_plan_with_search
:304

# Section 5: dfs_poi (Core DFS)
:884

# Section 6: Constraint Validation
:2672

# Section 7: Helper Methods
:2873

# Section 8: Delegation Methods
:3130
```

## 🔍 Key Methods

### High-Level Flow

```
run()
  └─> translate_nl2sl()         [if not oracle mode]
  └─> symbolic_search()
      └─> generate_plan_with_search()
          └─> dfs_poi()          [recursive, main search logic]
              └─> constraints_validation()
```

### Section 4: `generate_plan_with_search()` (Lines 304-883)

**Purpose**: High-level search orchestration

**Key Steps**:
1. Initialize search state and counters
2. Extract and assign user constraints
3. Collect intercity transport options (train/airplane)
4. Rank hotels and transports
5. Iterate through transport and hotel combinations
6. Call `dfs_poi()` for detailed POI planning
7. Return best plan or partial plan

**Calls**: `dfs_poi()`, constraint parsers, ranking engines

### Section 5: `dfs_poi()` (Lines 884-2671, 1765 lines)

**Purpose**: Core DFS search algorithm for POI planning

**Key Features**:
- Recursive depth-first search
- Day-by-day, activity-by-activity planning
- Handles multiple POI types: breakfast, lunch, dinner, attractions, hotels
- Applies user constraints (must-visit, must-not-visit, budget limits)
- Backtracking when constraints violated
- Returns first valid complete plan

**Decision Points**:
1. **First day**: Add go-transport
2. **Morning (00:00)**: Add breakfast at hotel
3. **Throughout day**: Select attractions, lunch, dinner based on time
4. **Evening**: Check for hotel or return transport
5. **Last day**: Add back-transport and validate

**Calls**: `add_*()` methods, `constraints_validation()`, `check_requirement()`, etc.

### Section 6: Constraint Validation (Lines 2672-2872)

**Methods**:
- `check_constraint()`: Budget constraints (attraction, restaurant, hotel, intercity, overall)
- `check_requirement()`: Must-visit POI requirements
- `check_budgets()`: Overall budget validation
- `constraints_validation()`: Complete plan validation (schema, commonsense, logic)
- `check_if_too_late()`: Time feasibility check

### Section 8: Delegation Methods (Lines 3130-3213)

**⚠️ IMPORTANT: DO NOT REMOVE THESE METHODS!**

These methods provide clean interface to modular components and are called throughout the search logic.

**Delegation targets**:
- **ConstraintParser**: `extract_user_constraints_by_DSL()`
- **RankingEngine**: `ranking_*()`, `calculate_distance()`, `get_transport_by_distance()`
- **POIManager**: `collect_poi_info_all()`, `collect_*_transport()`
- **PlanBuilder**: `add_*()` methods (intercity_transport, poi, accommodation, restaurant, attraction)

## 🏗️ Architecture

### Modular Components

```python
UrbanTrip (Main Agent)
    ├─> POIManager           # Collect POI data
    ├─> RankingEngine        # Rank POIs/transports
    ├─> ConstraintParser     # Parse user constraints
    ├─> ConstraintValidator  # Validate plans
    └─> PlanBuilder          # Build plan JSON
```

### Search Flow

```
1. NL2SL Translation (if needed)
   └─> LLM translates natural language to Python DSL

2. POI Collection
   └─> Collect all accommodations, attractions, restaurants

3. Constraint Parsing
   └─> Extract must-visit, must-not-visit, budgets, etc.

4. DFS Search
   ├─> Iterate intercity transports (train/airplane)
   ├─> Iterate hotels
   └─> For each combination:
       └─> dfs_poi() recursively builds daily itinerary
           ├─> Add breakfast
           ├─> Add attractions (with constraints)
           ├─> Add lunch/dinner (with constraints)
           ├─> Add hotel or return transport
           └─> Validate complete plan

5. Constraint Validation
   ├─> Schema validation (format, POI existence)
   ├─> Commonsense validation (time, budget, distance)
   └─> Logical validation (execute hard_logic_py)

6. Return Result
   └─> First valid plan or best partial plan
```

## 🎯 Usage Example

```python
from chinatravel.agent.load_model import init_agent, init_llm
from chinatravel.environment.world_env import WorldEnv

# Initialize
kwargs = {
    'method': 'UrbanTrip',
    'env': WorldEnv(),
    'backbone_llm': init_llm('glm-4.6'),
    'cache_dir': 'cache/',
    'debug': True,
}
agent = init_agent(kwargs)

# Run
query = {
    'uid': 'example_001',
    'nature_language': '我想去杭州玩两天...',
    'start_city': '上海',
    'target_city': '杭州',
    'days': 2,
    'people_number': 2,
    'hard_logic_py': [...]  # Parsed constraints
}

success, plan = agent.run(
    query,
    prob_idx='example_001',
    oralce_translation=True,  # Use ground truth constraints
    load_cache=True
)
```

## 📊 Output Format

```json
{
    "people_number": 2,
    "start_city": "上海",
    "target_city": "杭州",
    "itinerary": [
        {
            "day": 1,
            "activities": [
                {"type": "train", "start_time": "08:00", ...},
                {"type": "attraction", "position": "西湖", ...},
                {"type": "lunch", "position": "楼外楼", ...},
                {"type": "attraction", "position": "灵隐寺", ...},
                {"type": "dinner", "position": "知味观", ...},
                {"type": "accommodation", "position": "西湖酒店", ...}
            ]
        },
        {
            "day": 2,
            "activities": [...]
        }
    ],
    "input_token_count": 5234,
    "output_token_count": 312,
    "total_time": 12.5,
    "search_nodes": 456,
    "backtrack_count": 23,
    "all_constraints_pass": 1
}
```

## 🔧 Debugging Tips

### Enable Debug Output

```python
agent = init_agent({'debug': True, ...})
```

This will print detailed logs to:
- `cache/UrbanTrip_<llm>_<mode>/<uid>.log`
- `cache/UrbanTrip_<llm>_<mode>/<uid>.error`

### Common Issues

1. **No plan found**
   - Check constraint feasibility
   - Look for "backtrack..." messages in logs
   - Reduce constraint strictness

2. **Search too slow**
   - Reduce search space (fewer hotels/transports)
   - Increase `TIME_CUT` (default: 290 seconds)
   - Use oracle translation mode

3. **Constraints not satisfied**
   - Check `hard_logic_py` syntax
   - Verify POI names exist in database
   - Check budget limits are reasonable

## 📝 Development Notes

### Why is the file so large?

The `dfs_poi()` method (1765 lines) handles all the complex decision-making for POI selection. It's large because:

1. **Multiple POI types**: breakfast, lunch, dinner, attractions, hotels
2. **Complex constraints**: must-visit, must-not-visit, time windows, budgets
3. **Backtracking logic**: Need to try alternatives when constraints fail
4. **Inline optimization**: Many operations inlined for performance

### Future Refactoring Ideas

If you want to refactor in the future:

1. Extract POI selection logic into separate classes (AttractionSelector, RestaurantSelector, etc.)
2. Use Strategy pattern for different selection strategies
3. Move backtracking logic into a separate SearchState class
4. BUT: Ensure extensive testing before refactoring!

### Testing

Before making changes:

```bash
# Test basic functionality
python run_exp.py --splits easy_100 --agent UrbanTrip \
    --llm glm-4.6 --output_dir test_run

# Check output format
cat results/test_run/<uid>.json | jq .

# Verify logs
tail -100 cache/UrbanTrip_glm-4.6/test_run/<uid>.log
```

## 📚 Related Files

- `constraint_parser.py`: DSL parsing logic
- `constraint_validator.py`: Three-tier validation (schema, commonsense, logic)
- `ranking_engine.py`: POI and transport ranking algorithms
- `poi_manager.py`: POI data collection from environment
- `plan_builder.py`: JSON plan construction

---

**Last Updated**: 2024-11-18
**Maintainer**: Your Name
**Version**: 1.0 (Documented)
