import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.system.exitProcess

// Transformer Configuration class
class Config(
    val dim: Int, // transformer dimension
    val hiddenDim: Int, // for ffn layers
    val nLayers: Int, // number of layers
    val nHeads: Int, // number of query heads
    val nKvHeads: Int, // number of key/value heads (can be < query heads because of multiquery)
    val vocabSize: Int, // vocabulary size, usually 256 (byte-level)
    val seqLen: Int, // max sequence length
) {
    val sharedWeights: Boolean = vocabSize > 0
    val headSize: Int = dim / nHeads

    companion object {
        fun from(buffer: ByteBuffer): Config {
            val dim = buffer.int
            val hiddenDim = buffer.int
            val nLayers = buffer.int
            val nHeads = buffer.int
            val nKvHeads = buffer.int
            var vocabSize = buffer.int
            vocabSize = abs(vocabSize)
            val seqLen = buffer.int
            return Config(dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize, seqLen)
        }
    }
}

class Weights(
    val tokenEmbeddingTable: FloatArray, // (vocab_size, dim)
    val rmsAttWeight: FloatArray, // (layer, dim) rmsnorm weights
    val rmsFfnWeight: FloatArray, // (layer, dim)
    val wq: FloatArray, // (layer, dim, dim)
    val wk: FloatArray, // (layer, dim, dim)
    val wv: FloatArray, // (layer, dim, dim)
    val wo: FloatArray, // (layer, dim, dim)
    val w1: FloatArray, // (layer, hidden_dim, dim)
    val w2: FloatArray, // (layer, dim, hidden_dim)
    val w3: FloatArray, // (layer, hidden_dim, dim)
    val rmsFinalWeight: FloatArray, // (dim,)
    val freqCisReal: FloatArray, // (seq_len, dim/2)
    val freqCisImag: FloatArray, // (seq_len, dim/2)
    val wcls: FloatArray?, // (optional) classifier weights for the logits, on the last layer
) {
    companion object {

        fun from(config: Config, buffer: FloatBuffer): Weights {
            val tokenEmbeddingTable = array(buffer, config.vocabSize * config.dim)
            return Weights(
                tokenEmbeddingTable = tokenEmbeddingTable,
                rmsAttWeight = array(buffer, config.nLayers * config.dim),
                wq = array(buffer, config.nLayers * config.dim * config.dim),
                wk = array(buffer, config.nLayers * config.dim * config.dim),
                wv = array(buffer, config.nLayers * config.dim * config.dim),
                wo = array(buffer, config.nLayers * config.dim * config.dim),
                rmsFfnWeight = array(buffer, config.nLayers * config.dim),
                w1 = array(buffer, config.nLayers * config.hiddenDim * config.dim),
                w2 = array(buffer, config.nLayers * config.dim * config.hiddenDim),
                w3 = array(buffer, config.nLayers * config.hiddenDim * config.dim),
                rmsFinalWeight = array(buffer, config.dim),
                freqCisReal = array(buffer, config.seqLen * config.headSize / 2),
                freqCisImag = array(buffer, config.seqLen * config.headSize / 2),
                wcls = if (config.sharedWeights) tokenEmbeddingTable else null
            )
        }

        private fun array(buffer: FloatBuffer, size: Int): FloatArray {
            val floats = FloatArray(size)
            buffer[floats]
            return floats
        }
    }
}

// Run State class
class RunState(config: Config) {
    val x = FloatArray(config.dim)
    val xb = FloatArray(config.dim)
    val xb2 = FloatArray(config.dim)
    val hb = FloatArray(config.hiddenDim)
    val hb2 = FloatArray(config.hiddenDim)
    val q = FloatArray(config.dim)
    val k = FloatArray(config.dim)
    val v = FloatArray(config.dim)
    val att = FloatArray(config.nHeads * config.seqLen)
    val logits = FloatArray(config.vocabSize)
    val keyCache = FloatArray(config.nLayers * config.seqLen * config.dim)
    val valueCache = FloatArray(config.nLayers * config.seqLen * config.dim)

    init {
        // Ensure all memory allocations went fine
        if (x.isEmpty() || xb.isEmpty() || xb2.isEmpty() || hb.isEmpty() || hb2.isEmpty() || q.isEmpty() ||
            k.isEmpty() || v.isEmpty() || att.isEmpty() || logits.isEmpty() || keyCache.isEmpty() ||
            valueCache.isEmpty()
        ) {
            println("Memory allocation failed!")
            exitProcess(1)
        }
    }
}

object Llama2 {
    // ----------------------------------------------------------------------------
    // initialization: read from checkpoint
    // ----------------------------------------------------------------------------
    // neural net blocks
    private fun accum(a: FloatArray, b: FloatArray, size: Int) {
        for (i in 0 until size) {
            a[i] += b[i]
        }
    }

    private fun rmsNorm(o: FloatArray, x: FloatArray, weight: FloatArray, weightOffset: Int, size: Int) {
        // calculate sum of squares
        var ss = 0.0f
        for (j in 0 until size) {
            ss += x[j] * x[j]
        }
        ss /= size.toFloat()
        ss += 1e-5f
        ss = 1.0f / sqrt(ss.toDouble()).toFloat()
        // normalize and scale
        for (j in 0 until size) {
            o[j] = weight[weightOffset + j] * (ss * x[j])
        }
    }

    private fun softmax(x: FloatArray, xOffset: Int, size: Int) {
        // find max value (for numerical stability)
        var maxVal = x[0 + xOffset]
        for (i in 1 until size) {
            if (x[i + xOffset] > maxVal) {
                maxVal = x[i + xOffset]
            }
        }
        // exp and sum
        var sum = 0.0f
        for (i in 0 until size) {
            x[i + xOffset] = exp((x[i + xOffset] - maxVal).toDouble()).toFloat()
            sum += x[i + xOffset]
        }
        // normalize
        for (i in 0 until size) {
            x[i + xOffset] /= sum
        }
    }

    private fun matmul(xout: FloatArray, x: FloatArray, w: FloatArray?, wOffset: Int, n: Int, d: Int) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        for (i in 0 until d) {
            var value = 0.0f
            for (j in 0 until n) {
                value += w!![wOffset + i * n + j] * x[j]
            }
            xout[i] = value
        }
    }

    private fun transformer(token: Int, pos: Int, p: Config, s: RunState, w: Weights) {
        // a few convenience variables
        val dim = p.dim
        val hiddenDim = p.hiddenDim
        val headSize = p.headSize

        // copy the token embedding into x
        System.arraycopy(w.tokenEmbeddingTable, token * dim, s.x, 0, dim)

        // forward all the layers
        for (l in 0 until p.nLayers) {

            // attention rmsnorm
            rmsNorm(s.xb, s.x, w.rmsAttWeight, dim * l, dim)

            // qkv matmuls for this position
            matmul(s.q, s.xb, w.wq, dim * dim * l, dim, dim)
            matmul(s.k, s.xb, w.wk, dim * dim * l, dim, dim)
            matmul(s.v, s.xb, w.wv, dim * dim * l, dim, dim)

            // apply RoPE rotation to the q and k vectors for each head
            for (h in 0 until p.nHeads) {
                // get the q and k vectors for this head
                val qOffset = h * headSize
                val kOffset = h * headSize
                // float* q = s->q + h * head_size;
                // float* k = s->k + h * head_size;
                // rotate q and k by the freq_cis_real and freq_cis_imag
                var i = 0
                while (i < headSize) {
                    val q0 = s.q[qOffset + i]
                    val q1 = s.q[qOffset + i + 1]
                    val k0 = s.k[kOffset + i]
                    val k1 = s.k[kOffset + i + 1]
                    val fcr = w.freqCisReal[pos * headSize / 2 + i / 2]
                    val fci = w.freqCisImag[pos * headSize / 2 + i / 2]
                    s.q[qOffset + i] = q0 * fcr - q1 * fci
                    s.q[qOffset + i + 1] = q0 * fci + q1 * fcr
                    s.k[kOffset + i] = k0 * fcr - k1 * fci
                    s.k[kOffset + i + 1] = k0 * fci + k1 * fcr
                    i += 2
                }
            }

            // save key,value at this time step (pos) to our kv cache
            val loff = l * p.seqLen * dim // kv cache layer offset for convenience
            System.arraycopy(s.k, 0, s.keyCache, loff + pos * dim, dim)
            System.arraycopy(s.v, 0, s.valueCache, loff + pos * dim, dim)

            // multihead attention. iterate over all heads
            for (h in 0 until p.nHeads) {
                // get the query vector for this head
                // float* q = s.q + h * head_size;
                val qOffset = h * headSize

                // attention scores for this head
                // float* att = s.att + h * p.seq_len;
                val attOffset = h * p.seqLen

                // iterate over all timesteps, including the current one
                for (t in 0..pos) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * head_size;
                    val keyCacheOffset = loff + t * dim + h * headSize
                    // calculate the attention score as the dot product of q and k
                    var score = 0.0f
                    for (i in 0 until headSize) {
                        score += s.q[qOffset + i] * s.keyCache[keyCacheOffset + i]
                    }
                    score /= sqrt(headSize.toDouble()).toFloat()
                    // save the score to the attention buffer
                    s.att[attOffset + t] = score
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(s.att, attOffset, pos + 1)

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * head_size;
                val xbOffset = h * headSize
                // memset(xb, 0, head_size * sizeof(float));
                for (i in 0 until headSize) {
                    s.xb[xbOffset + i] = 0f
                }
                for (t in 0..pos) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * head_size;
                    val vOffset = loff + t * dim + h * headSize
                    // get the attention weight for this timestep
                    val a = s.att[attOffset + t]
                    // accumulate the weighted value inconfigto xb
                    for (i in 0 until headSize) {
                        s.xb[xbOffset + i] += a * s.valueCache[vOffset + i]
                    }
                }
            }

            // final matmul to get the output of the attention
            matmul(s.xb2, s.xb, w.wo, dim * dim * l, dim, dim)

            // residual connection back into x
            accum(s.x, s.xb2, dim)

            // ffn rmsnorm
            rmsNorm(s.xb, s.x, w.rmsFfnWeight, dim * l, dim)

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(s.hb, s.xb, w.w1, dim * p.hiddenDim * l, dim, p.hiddenDim)
            matmul(s.hb2, s.xb, w.w3, p.hiddenDim * dim * l, dim, p.hiddenDim)

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for (i in 0 until hiddenDim) {
                s.hb[i] = s.hb[i] / (1.0f + exp(-s.hb[i].toDouble()).toFloat())
            }

            // elementwise multiply with w3(x)
            for (i in 0 until hiddenDim) {
                s.hb[i] = s.hb[i] * s.hb2[i]
            }

            // final matmul to get the output of the ffn
            // matmul(s.xb, s.hb, w.w2 + l*dim*hidden_dim, hidden_dim, dim);
            matmul(s.xb, s.hb, w.w2, dim * p.hiddenDim * l, p.hiddenDim, dim)

            // residual connection
            accum(s.x, s.xb, dim)
        }

        // final rmsnorm
        rmsNorm(s.x, s.x, w.rmsFinalWeight, 0, dim)

        // classifier into logits
        matmul(s.logits, s.x, w.wcls, 0, dim, p.vocabSize)
    }

    // ----------------------------------------------------------------------------
    // byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt
    private fun strLookup(str: String, vocab: Array<String?>, vocabSize: Int): Int {
        // find the first perfect match for str in vocab, return its index or -1 if not found
        for (i in 0 until vocabSize) {
            if (str == vocab[i]) {
                return i
            }
        }
        return -1
    }

    private fun bpeEncode(
        text: String,
        vocab: Array<String?>,
        vocabScores: FloatArray,
        vocabSize: Int,
        tokens: IntArray,
    ): Int {
        // first encode every individual byte in the input string
        var nTokens = 0 // the number of tokens
        for (element in text) {
            val singleChar = element.toString()
            val id = strLookup(singleChar, vocab, vocabSize)
            if (id == -1) {
                System.out.printf("not good\n")
                System.exit(1)
            }
            tokens[nTokens] = id
            nTokens++
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            var bestScore = -1e10f
            var bestId = -1
            var bestIdx = -1
            for (i in 0 until nTokens - 1) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                val strBuffer = vocab[tokens[i]] + vocab[tokens[i + 1]]
                // sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
                val id = strLookup(strBuffer, vocab, vocabSize)
                if (id != -1 && vocabScores[id] > bestScore) {
                    // this merge pair exists in vocab! record its score and position
                    bestScore = vocabScores[id]
                    bestId = id
                    bestIdx = i
                }
            }
            if (bestIdx == -1) {
                break // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[bestIdx] = bestId
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (i in bestIdx + 1 until nTokens - 1) {
                tokens[i] = tokens[i + 1]
            }
            nTokens-- // token length decreased
        }
        return nTokens
    }

    // ----------------------------------------------------------------------------
    // utilities
    private fun timeInMs(): Long {
        // return time in milliseconds, for benchmarking the model speed
        return System.nanoTime() / 1000000
    }

    private var rngSeed: Long = 0
    private fun randomU32(): Int {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        rngSeed = rngSeed xor (rngSeed shr 12)
        rngSeed = rngSeed xor (rngSeed shl 25)
        rngSeed = rngSeed xor (rngSeed shr 27)
        return (rngSeed * 0x2545F4914F6CDD1DL shr 32).toInt()
    }

    private fun randomF32(): Float { // random float32 in [0,1)
        return (randomU32() ushr 8) / 16777216.0f
    }

    private fun sample(probabilities: FloatArray, n: Int): Int {
        // sample index from probabilities, they must sum to 1
        val r = randomF32()
        var cdf = 0.0f
        for (i in 0 until n) {
            cdf += probabilities[i]
            if (r < cdf) {
                return i
            }
        }
        return n - 1 // in case of rounding errors
    }

    private fun argmax(v: FloatArray, n: Int): Int {
        // return argmax of v in elements 0..n
        var maxI = 0
        var maxP = v[0]
        for (i in 1 until n) {
            if (v[i] > maxP) {
                maxI = i
                maxP = v[i]
            }
        }
        return maxI
    }

    // ----------------------------------------------------------------------------
    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {

        // poor man's C argparse
        var checkPoint: String? = null // e.g. out/model.bin
        var temperature = 0.0f // 0.9f; // e.g. 1.0, or 0.0
        var steps = 256 // max number of steps to run for, 0: use seq_len
        var prompt: String? = null // prompt string

        // 'checkpoint' is necessary arg
        if (args.isEmpty()) {
            println("Usage: java -jar Llama2.jar <checkpoint_file> [temperature] [steps] [prompt]\n")
            System.exit(1)
        }
        if (args.isNotEmpty()) {
            checkPoint = args[0]
        }
        if (args.size >= 2) {
            // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
            temperature = args[1].toFloat()
        }
        if (args.size >= 3) {
            steps = args[2].toInt()
        }
        if (args.size >= 4) {
            prompt = args[3]
        }

        // seed rng with time. if you want deterministic behavior use temperature 0.0
        rngSeed = System.currentTimeMillis() / 1000 // (unsigned int)time(NULL);
        var config: Config
        var weights: Weights
        FileChannel.open(Paths.get(checkPoint), StandardOpenOption.READ).use { fileChannel ->
            val bb: ByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size())
            bb.order(ByteOrder.LITTLE_ENDIAN)
            // read in the config header
            config = Config.from(bb)
            weights = Weights.from(config, bb.asFloatBuffer())
        }

        // right now we cannot run for more than config.seq_len steps
        if (steps <= 0 || steps > config.seqLen) {
            steps = config.seqLen
        }

        // read in the tokenizer.bin file
        val vocab = arrayOfNulls<String>(config.vocabSize)
        val vocabScores = FloatArray(config.vocabSize)
        FileChannel.open(Paths.get("tokenizer.bin"), StandardOpenOption.READ).use { channel ->
            val bb: ByteBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
            bb.order(ByteOrder.LITTLE_ENDIAN)
            val maxTokenLength = bb.getInt()
            for (i in 0 until config.vocabSize) {
                vocabScores[i] = bb.getFloat()
                val len = bb.getInt()
                val bytes = ByteArray(len)
                bb[bytes]
                vocab[i] = String(bytes)
            }
        }

        val state = RunState(config)

        // process the prompt, if any
        var promptTokens: IntArray? = null
        var numPromptTokens = 0
        if (prompt != null) {
            promptTokens = IntArray(config.seqLen)
            numPromptTokens = bpeEncode(prompt, vocab, vocabScores, config.vocabSize, promptTokens)
        }

        // start the main loop
        var start: Long = 0 // used to time our code, only initialized after first iteration
        var next: Int // will store the next token in the sequence
        var token = 1 // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        var pos = 0 // position in the sequence
        println("<s>") // explicit print the initial BOS token for stylistic symmetry
        // reasons
        while (pos < steps) {

            // forward the transformer to get logits for the next token
            transformer(token, pos, config, state, weights)
            if (pos < numPromptTokens) {
                // if we are still processing the input prompt, force the next prompt token
                next = promptTokens!![pos]
            } else {
                // sample the next token
                if (temperature == 0.0f) {
                    // greedy argmax sampling: take the token with the highest probability
                    next = argmax(state.logits, config.vocabSize)
                } else {
                    // apply the temperature to the logits
                    for (q in 0 until config.vocabSize) {
                        state.logits[q] /= temperature
                    }
                    // apply softmax to the logits to get the probabilities for next token
                    softmax(state.logits, 0, config.vocabSize)
                    // we sample from this distribution to get the next token
                    next = sample(state.logits, config.vocabSize)
                }
            }

            // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR#89)
            val tokenStr = if (token == 1 && vocab[next]!![0] == ' ') vocab[next]!!.substring(1) else vocab[next]!!
            System.out.printf("%s", tokenStr)
            System.out.flush()

            // advance forward
            token = next
            pos++
            // init our timer here because the first iteration is slow due to memmap
            if (start == 0L) {
                start = timeInMs()
            }
        }

        // report achieved tok/s
        val end = timeInMs()
        System.out.printf("\nachieved tok/s: %f\n", (steps - 1) / (end - start).toDouble() * 1000)
    }
}
