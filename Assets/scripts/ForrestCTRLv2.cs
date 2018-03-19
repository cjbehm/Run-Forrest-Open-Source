using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using NeuralNet;

[RequireComponent(typeof(Rigidbody))]
public class ForrestCTRLv2 : ForrestCTRL {
    public float adventurousness;
    public float speed;
    public float acceleration;
    public float totalDistance;
    int updates;
    public float averageSpeed { get { return totalDistance / updates; } }

    public Dictionary<Vector2, int>memory;

    // Use this for initialization
    override protected void Start() {
        memory = new Dictionary<Vector2, int>();
        updates = 0;
        speed = 10;
        acceleration = 0;
        totalDistance = 0;
        base.Start();
/*        menu = SceneManager.GetActiveScene().name == "menu";

        if (!menu)
        {
            // Grab reference to the controller script
            C = Camera.main.GetComponent<ctrl>();

            // Sets the max lap according to the controller
            lap.y = C.mLaps;
        }
        else
        {
            nn.IniWeights(new float[] { 3.472079f, 1.762525f, -2.266208f, 0.8920379f, -3.915989f, -1.762377f, -2.844904f, 3.381477f, 1.12464f, -3.086241f, 3.320154f, 0.1941123f, 0.1791953f, -3.122393f, 0.8971314f, 0.1158746f, 3.512217f, 1.440832f, 3.3429f, -3.377463f, -2.171291f, 1.523072f, -2.242229f, -2.650826f, 3.01321f, -3.341551f, 3.746894f, -1.755286f, -0.3875917f });
        }

        // Grab ref to ridgid body component
        rb = GetComponent<Rigidbody>();

        // Set rotation constraints so Forrest doesn't fall over
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        // Grab ref to collider
        col = GetComponent<Collider>();

        // 
        tag = "Active";

        // Set the laps to 1 instead of 0 at every start no matter what
        lap.x = 1;*/
    }

    // Update is called once per frame
    override protected void FixedUpdate()
    {
        if(nn.inputs < 1) { Debug.Log("Forrest doesn't have a brain"); return; }

        updates += 1;
        float[] nnOutputs;
        speed = speed + acceleration;
        if(speed > 15) { speed = 15; }
        //if(speed < 5) { speed = 5; }
        totalDistance += speed;

        if(speed < 5.0f) {
            Freeze();
        }

        // Set up a raycast hit for knowing what we hit
        RaycastHit hit;

        // Set up out 5 feelers for undertanding the world
        Vector3[] feeler = new Vector3[]
        {
            // 0 = L
            transform.TransformDirection(Vector3.left),
            // 1 - FL
            transform.TransformDirection(Vector3.left+Vector3.forward),
            // 2 - F
            transform.TransformDirection(Vector3.forward),
            // 3 = FR
            transform.TransformDirection(Vector3.right + Vector3.forward),
            // 4 = R
            transform.TransformDirection(Vector3.right),
        };

        // Use this to collect all feeler distances, then well pass them through our NN for an output
        inp = new float[feeler.Length+2];

        brain = nn.ReadBrain();

        // Rotate the charater based on Horizonal Input & later NN Output
        transform.rotation = Quaternion.Euler(transform.eulerAngles + Vector3.up * movement * 2.5f);

        // remember where we've already been
        Vector3 nextPos = transform.position + transform.forward * (Time.fixedDeltaTime * speed);
        Vector2 curPos = new Vector2(nextPos.x,nextPos.z);
        if(Visited(curPos,Time.fixedDeltaTime*(speed - 0.1f)))
        {
            inp[inp.Length-2] = 0.0f;
        }
        else
        {
            inp[inp.Length-2] = adventurousness;
            if(!memory.ContainsKey(curPos)) {
                memory.Add(curPos, 1);
            }
        }

        // If attempt has ended
        if (!ended)
        {
            // Auto move Forrest forward
            base.rb.MovePosition(transform.position + transform.forward * (Time.fixedDeltaTime * speed));
        }

        // Loop through all feelers
        for (int i = 0; i < feeler.Length; i++)
        {
            inp[i] = 0;
            // See what all feelers feel
            if (Physics.Raycast(transform.position, feeler[i], out hit))
            {
                // If feelers feel something other than Forrest & nothing
                if (hit.collider != null && hit.collider != col)
                {
                    // Set the input[i] to be the distance of feeler[i]
                    inp[i] = hit.distance;

                    // Draw the feelers in the Scene mode
                    Debug.DrawRay(transform.position, feeler[i] * hit.distance, Color.red);
                }

            }

            
        }
        inp[inp.Length - 1] = speed;
        // Add to our fitness every frame
        fitness += (ended) ? 0 : inp2fit(inp);

        nnOutputs = nn.CalculateNN(inp);
        acceleration = nnOutputs[1] * 0.01f;
        adventurousness = nnOutputs[2];
        // This sets the output text display to be the output of our NN
        if (!menu)
        {
            movement = ended ? 0 : ((C.intelli == ctrl.IntelMode.Human) ? Input.GetAxis("Horizontal") : nnOutputs[0]);
        }
        else
        {
            movement = ended ? 0 : (nnOutputs[0]);
        }

        // 
        if (!menu && !ended && lap.x>lap.y)
        {
            Freeze();
            CheckIfLast();
        }
    }

    /// <summary>
    /// This converts your inputs to a fitness value
    /// </summary>
    /// <param name="inps"></param>
    /// <returns></returns>
    float inp2fit(float[] inps)
    {
        float ret = 0;

        // 
        for (int i = 0; i < inps.Length; i++)
        {
            ret += inps[i];
        }
        ret += speed;
        return ((ret/inps.Length+1)/100)*lap.x;
    }

    override public void Reset()
    {
        memory = new Dictionary<Vector2, int>();//new HashSet<Vector2>();
        speed = 10;
        totalDistance = 0;
        acceleration = 0;
        updates = 0;
        base.Reset();
    }

    // return true if position is within threshold distance of a
    // location that we have already visited
    bool Visited(Vector2 position, float threshold = 0.0f) {
        // optimisation if the threshold is 0, then this is just containsKey
        if (threshold == 0.0f)
        {
            return memory.ContainsKey(position);
        }
        else
        {
            float dist;
            foreach(KeyValuePair<Vector2, int> entry in memory) {
                dist = Vector2.Distance(entry.Key,position);
                // ignore entries that are beyond the threshold
                if (dist <= threshold) {
                    return true;
                }
            }
        }
        return false;
    }
}
