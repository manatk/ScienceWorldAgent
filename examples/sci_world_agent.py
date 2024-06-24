import time
import random
import argparse
import json
from typing import Dict, List


from scienceworld import ScienceWorldEnv
from openai import OpenAI

client = OpenAI()

from openai.types.chat.chat_completion import ChatCompletion

def generate_text(prompt: str, max_tokens: int = 4096, temperature: float = 0.5) -> str:
    response = client.chat.completions.create(model="gpt-4-turbo",
    messages=[
            {"role": "system", "content": "You are a science experimenter who is trying to complete the task as confidently as possible."},
            {"role": "user", "content": prompt}
        ],    
    max_tokens=max_tokens,
    temperature=temperature)
    return response.choices[0].message.content
    #breakpoint()
    return response.choices[0].text.strip()

def planSteps(taskDescription, observations, actionsTaken):
    prompt = "You are a science experimenter who is trying to complete the task as confidently as possible. THIS IS YOUR GOAL. EVERY STEP YOU TAKE MUST BRING YOU CLOSER TO YOUR GOAL: " + taskDescription
    prompt += f"Here are the recent environment states including the most recent state: {observations}"
    prompt += f"Here are the actions you have taken so far: {actionsTaken}."
    objectPrompt = f"Before you take more actions, spend some time planning about what to do next. Specifically consider: what objects do you need and where will you locate them? List objects and their corresponding locations one by one."
    objectsNeeded = generate_text(prompt + objectPrompt)
    
    objectCollectPrompt = f"Here are the objects you listed" + objectsNeeded + "are there any objects which you have not yet collected?"
    objectsToCollect = generate_text(objectCollectPrompt)

    subgoalsPrompt = f"Next, considering the task, what are important subgoals that we need to achieve? List out each subgoal one by one."
    subgoals = generate_text(prompt + subgoalsPrompt)

    progressPrompt = f"Here are the subgoals you listed: " + subgoals + "What subgoals have yet to be completed?"
    progress = generate_text(progressPrompt)

    returnString = "Here is the plan you derived. These are the objects that you need to complete the task" + objectsNeeded + "Amongst these, you have yet to collect these objects: " + objectsToCollect + "These are the subgoals you listed out: " + subgoals + "Here are the subgoals that have yet to be completed: " + progress
    return returnString

def sciWorldAgent(args):
    """ Example random agent -- randomly picks an action at each step. """
    exitCommands = ["quit", "exit"]

    taskIdx = args['task_num']
    simplificationStr = args['simplification_str']
    numEpisodes = args['num_episodes']

    # Keep track of the agent's final scores
    finalScores = []

    # Initialize environment
    env = ScienceWorldEnv("", args['jar_path'], envStepLimit=args['env_step_limit'])

    taskNames = env.get_task_names()
    print("Task Names: " + str(taskNames))

    # Choose task
    taskName = taskNames[taskIdx]        # Just get first task
    # Load the task, we we have access to some extra accessors e.g. get_random_variation_train()
    env.load(taskName, 0, "")
    maxVariations = env.get_max_variations(taskName)
    print("Starting Task " + str(taskIdx) + ": " + taskName)
    time.sleep(2)

    # Start running episodes
    for episodeIdx in range(0, numEpisodes):
        # Pick a random task variation
        randVariationIdx = env.get_random_variation_train()
        env.load(taskName, randVariationIdx, simplificationStr)

        # Reset the environment
        initialObs, initialDict = env.reset()

        # Example accessors
        # print("Possible actions: " + str(env.get_possible_actions()))
        # print("Possible objects: " + str(env.get_possible_objects()))
        templates, lut = env.get_possible_action_object_combinations()
        # print("Possible action/object combinations: " + str(templates))
        # print("Object IDX to Object Referent LUT: " + str(lut))
        print("Task Name: " + taskName)
        print("Task Variation: " + str(randVariationIdx) + " / " + str(maxVariations))
        print("Task Description: " + str(env.get_task_description()))
        taskDescription = env.get_task_description()
        print("look: " + str(env.look()))
        print("inventory: " + str(env.inventory()))
        print("taskdescription: " + str(env.taskdescription()))

        score = 0.0
        isCompleted = False
        curIter = 0
        maxScore = 0
        # Run one episode until we reach a stopping condition (including exceeding the maximum steps)
        userInputStr = "look around"        # First action
        actionsTaken = []
        observations = []

        plan = planSteps(taskDescription, observations, actionsTaken)
        print(plan)

        while (userInputStr not in exitCommands) and (isCompleted is False) and (curIter < 100):


            print("----------------------------------------------------------------")
            print("Step: " + str(curIter))

            # Send user input, get response
            print(">>> " + userInputStr)
            
            observation, reward, isCompleted, info = env.step(userInputStr.strip())
            if observation != "No known action matches that input.":
                #print(validActions)
                observations.append(observation)
            score = info['score']
            maxScore = max(maxScore, score)

            print("\n>>> " + observation)
            print("Reward: " + str(reward))
            print("Score: " + str(score))
            print("isCompleted: " + str(isCompleted))

            # The environment will make isCompleted `True` when a stop condition
            # has happened, or the maximum number of steps is reached.
            if (isCompleted):
                break

            # Randomly select action

            # Any action (valid or not)
            # templates, lut = env.get_possible_action_object_combinations()
            # print("Possible action/object combinations: " + str(templates))
            # print("Object IDX to Object Referent LUT: " + str(lut))
            # randomTemplate = random.choice( templates )
            # print("Next random action: " + str(randomTemplate))
            # userInputStr = randomTemplate["action"]

            # Only valid actions
            validActions = env.get_valid_action_object_combinations_with_templates()
            randomAction = random.choice(validActions)
            # breakpoint()
            actions = [action["action"] for action in validActions]
            
            if curIter % 5 == 0:
                plan = planSteps(taskDescription, observations, actionsTaken)

            prompt = "THIS IS YOUR GOAL. EVERY STEP YOU TAKE MUST BRING YOU CLOSER TO YOUR GOAL: " + taskDescription
            prompt = "This is the plan which you derived" + plan
            prompt += f"Here are the recent environment states including the most recent state: {observations}"
            prompt += f"Here are the actions you have taken so far: {actionsTaken}."
            prompt +=f"You are forbidden from taking these actions again since you just took them:" + "".join(actionsTaken[-10:])
            prompt += f"Pick the best next step to take. YOU MUST CHOOSE FROM THE FOLLOWING: {actions}"
            

            prompt += f"RETURN JUST THE ACTION NO EXTRA TEXT OR PUNCTUATION. THE ACTION MUST BE FROM THE LIST."
            action = generate_text(prompt)
            actionsTaken.append(action)
            userInputStr = action.lower().strip()
            print(list(lut.keys())[-1])  
            print("Choosing action: " + str(userInputStr))

            # Keep track of the number of commands sent to the environment in this episode
            curIter += 1

        print("Goal Progress:")
        print(env.get_goal_progress())
        time.sleep(1)

        # Episode finished -- Record the final score
        print(score)
        finalScores.append(maxScore)

        # Report progress of model
        print("Final score: " + str(score))
        print("isCompleted: " + str(isCompleted))

        # Save history -- and when we reach maxPerFile, export them to file
        filenameOutPrefix = args['output_path_prefix'] + str(taskIdx)
        env.store_run_history(episodeIdx, notes={'text': 'my notes here'})
        env.save_run_histories_buffer_if_full(filenameOutPrefix, max_per_file=args['max_episode_per_file'])

    # Episodes are finished -- manually save any last histories still in the buffer
    env.save_run_histories_buffer_if_full(filenameOutPrefix, max_per_file=args['max_episode_per_file'], force_save=True)

    # Show final episode scores to user
    # Clip negative scores to 0 for average calculation
    print(finalScores)
    avg = sum([x for x in finalScores if x >= 0]) / len(finalScores)
    print("")
    print("---------------------------------------------------------------------")
    print(" Summary (Random Agent)")
    print(" Task " + str(taskIdx) + ": " + taskName)
    print(" Simplifications: " + str(simplificationStr))
    print("---------------------------------------------------------------------")
    print(" Episode scores: " + str(finalScores))
    print(" Average episode score: " + str(avg))
    print("---------------------------------------------------------------------")
    print("")

    print("Completed.")


def build_simplification_str(args):
    """ Build simplification_str from args. """
    simplifications = list()
    if args["teleport"]:
        simplifications.append("teleportAction")

    if args["self_watering_plants"]:
        simplifications.append("selfWateringFlowerPots")

    if args["open_containers"]:
        simplifications.append("openContainers")

    if args["open_doors"]:
        simplifications.append("openDoors")

    if args["no_electrical"]:
        simplifications.append("noElectricalAction")

    return args["simplifications_preset"] or ",".join(simplifications)


#
#   Parse command line arguments
#
def parse_args():
    desc = "Run a model that chooses random actions until successfully reaching the goal."
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("--jar_path", type=str,
                        help="Path to the ScienceWorld jar file. Default: use builtin.")
    parser.add_argument("--task-num", type=int, default=13,
                        help="Specify the task number to play. Default: %(default)s")
    parser.add_argument("--var-num", type=int, default=0,
                        help="Specify the task variation number to play. Default: %(default)s")
    parser.add_argument("--env-step-limit", type=int, default=100,
                        help="Maximum number of steps per episode. Default: %(default)s")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes to play. Default: %(default)s")
    parser.add_argument("--seed", type=int,
                        help="Seed the random generator used for sampling random actions.")
    parser.add_argument("--output-path-prefix", default="save-histories",
                        help="Path prefix to use for saving episode transcripts. Default: %(default)s")
    parser.add_argument("--max-episode-per-file", type=int, default=1000,
                        help="Maximum number of episodes per transcript file. Default: %(default)s")

    simplification_group = parser.add_argument_group('Game simplifications')
    simplification_group.add_argument("--simplifications-preset", choices=['easy'],
                                      help="Choose a preset among: 'easy' (apply all possible simplifications).")
    simplification_group.add_argument("--teleport", action="store_true",
                                      help="Lets agents instantly move to any location.")
    simplification_group.add_argument("--self-watering-plants", action="store_true",
                                      help="Plants do not have to be frequently watered.")
    simplification_group.add_argument("--open-containers", action="store_true",
                                      help="All containers are opened by default.")
    simplification_group.add_argument("--open-doors", action="store_true",
                                      help="All doors are opened by default.")
    simplification_group.add_argument("--no-electrical", action="store_true",
                                      help="Remove the electrical actions (reduces the size of the action space).")

    args = parser.parse_args()
    params = vars(args)
    return params


def main():
    print("ScienceWorld 1.0 API Examples - Random Agent")
    # Parse command line arguments
    args = parse_args()
    random.seed(args["seed"])
    args["simplification_str"] = build_simplification_str(args)
    sciWorldAgent(args)


if __name__ == "__main__":
    main()
