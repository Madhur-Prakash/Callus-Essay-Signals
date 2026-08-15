/**
 * Example essays for the "load an example" affordance.
 *
 * These exist so a first-time visitor can see the tool work without pasting
 * anything of their own. The labels describe *how each was produced*, not what
 * the detector will say about it - the detector's answer is whatever it measures,
 * and presenting an expected verdict would be pre-loading the conclusion.
 */

export interface ExampleEssay {
  id: string;
  name: string;
  provenance: string;
  text: string;
}

const HUMAN_STYLE = `The robot never worked. That is the honest summary of my sophomore year. I spent seven months on a line-following car that could not follow a line, and I want to explain why that matters to me.

My design was bad from the start. I used two cheap IR sensors mounted too close together, maybe four centimetres apart, because that was what fit on the breadboard I already owned. When the car hit a curve the sensors both read the same value and the controller just guessed. I didn't know the phrase "insufficient sensor separation" then. I only knew that my car drove into a wall in front of forty people at the regional meet, and that the kid from Central High laughed.

I rebuilt it. Not immediately, though. I put the whole thing in a shoebox under my bed for about six weeks and told my mom I was done with robotics. Then in January I got bored and pulled it out again, and this time I actually read the sensor datasheet instead of guessing. Four centimetres was wrong. Eleven was better. I also learned that my PID loop had no derivative term at all, which is a little embarrassing to admit in writing.

The car placed fourth in April. Not first. Fourth. But it finished the course three times out of three, and when it did I remember standing there with my hands in my pockets feeling something that was not exactly pride. It was more like relief, the specific relief of a thing that finally does what you told it to do. I still chase that feeling.`;

const MACHINE_STYLE = `From an early age, I have been drawn to robotics. What began as a modest interest gradually developed into a genuine commitment. Moreover, the environment in which I worked was demanding, and it required consistent effort. Furthermore, I approached the work methodically, building my understanding one step at a time.

The most significant challenge arose when my initial approach proved inadequate. Additionally, progress was neither linear nor guaranteed, and there were periods of real difficulty. Consequently, I encountered a setback that forced me to reconsider my fundamental assumptions. The obstacle was not merely technical but also personal, testing my resolve.

The turning point came when I decided to rebuild my approach from first principles. Moreover, recognising the limits of my method, I sought guidance and revised my strategy. It required patience, precision, and a willingness to fail. Ultimately, I began to document my process carefully, which transformed how I understood it.

The experience instilled in me a deeper appreciation for patience and iteration. Furthermore, I came to understand that meaningful progress depends on perseverance rather than talent. Ultimately, this process cultivated in me a durable capacity for intellectual humility. As I look toward university, I intend to bring this same commitment to engineering.`;

const MIXED_STYLE = `The robot never worked. That is the honest summary of my sophomore year. I spent seven months on a line-following car that could not follow a line, and I want to explain why that matters to me.

My initial design was fundamentally flawed. I employed two inexpensive infrared sensors mounted in close proximity, approximately four centimetres apart, a configuration dictated by the constraints of the breadboard I already possessed. Consequently, when the vehicle encountered a curve, both sensors registered identical values and the controller was left to estimate. Moreover, I was at that time unfamiliar with the concept of insufficient sensor separation, and I understood only that my vehicle had collided with a wall before an audience of forty individuals at the regional competition.

I rebuilt it. Not immediately, though. I put the whole thing in a shoebox under my bed for about six weeks and told my mom I was done with robotics. Then in January I got bored and pulled it out again, and this time I actually read the sensor datasheet instead of guessing. Four centimetres was wrong. Eleven was better.

The car placed fourth in April. Not first. Fourth. But it finished the course three times out of three, and when it did I remember standing there with my hands in my pockets feeling something that was not exactly pride. It was more like relief. I still chase that feeling.`;

export const EXAMPLE_ESSAYS: ExampleEssay[] = [
  {
    id: 'human-style',
    name: 'Hand-written draft',
    provenance:
      'A hand-authored seed essay from this project’s corpus - uneven sentence lengths, contractions, concrete detail.',
    text: HUMAN_STYLE,
  },
  {
    id: 'machine-style',
    name: 'Machine-register essay',
    provenance:
      'Produced by the offline template generator: even sentence lengths, dense formal connectives, abstract nouns.',
    text: MACHINE_STYLE,
  },
  {
    id: 'mixed-style',
    name: 'Human draft, one paragraph rewritten',
    provenance:
      'The hand-written draft with its second paragraph formalised - the localised-edit case the style-shift analysis targets.',
    text: MIXED_STYLE,
  },
];
