package vowpalWabbit.learner.concurrent;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

import vowpalWabbit.learner.VWLearner;
import vowpalWabbit.learner.VWLearners;
import vowpalWabbit.learner.VWMulticlassLearner;
import vowpalWabbit.learner.VWMultilabelsLearner;

/**
 * Factory responsible for creating a single concurrent predictor instance which behind the
 * scenes distributes the prediction work to a learner pool. One should ensure that number of threads
 * in the threadpool is at least as big as the pool size otherwise you are wasting memory occupied by the
 * extra learner instances.
 * 
 * @author atulvkamat
 * 
 */
public class VWConcurrentLearnerFactory {

    /**
     * Create a multilabel predictor that distributes the work of multilabel
     * prediction to backing thread pool specified as executor service. One can
     * use the predictor returned to predict with a timeout.
     * 
     * A fixed size of poolSize thread pool will be created.
     * 
     * @param predictorName
     *            name of the predictor
     * @param poolSize
     *            for predictor pool and thread pool created internally.
     * @param command
     *            command arguments as you would have used in command line for
     *            initializing VW
     * @return VWConcurrentMultilabelsPredictor instance.
     * @throws InterruptedException
     */
    public static VWConcurrentMultilabelsPredictor createMultilabelsPredictor(
            final String predictorName,
            final int poolSize,
            final String command) throws InterruptedException {
        return createMultilabelsPredictor(predictorName, Executors.newFixedThreadPool(poolSize), poolSize, command);
    }

    /**
     * Create a multilabels predictor that distributes the work of multilabel
     * prediction to backing thread pool specified as executor service. One can
     * use the predictor returned to predict with a timeout.
     * 
     * This method offers flexibility to reuse an existing thread pool for
     * executing the predictors. In an ideal case, the total number of
     * predictors should be no less than the number of threads to achieve full
     * utilization. Otherwise memories will be wasted for holding predictors
     * that are not utilized.
     * 
     * @param predictorName
     *            name of the predictor
     * @param learnerExecutor
     *            thread pool to be used
     * @param poolSize
     *            for predictor pool created internally.
     * @param command
     *            command arguments as you would have used in command line for
     *            initializing VW
     * @return VWConcurrentMultilabelsPredictor instance.
     * @throws InterruptedException
     */
    public static VWConcurrentMultilabelsPredictor createMultilabelsPredictor(
            final String predictorName,
            final ExecutorService learnerExecutor, final int poolSize,
            final String command) throws InterruptedException {

        final LinkedBlockingQueue<VWMultilabelsLearner> vwPredictorPool = createVWPredictorPool(
                learnerExecutor, poolSize, command);
        return new VWConcurrentMultilabelsPredictor(predictorName, learnerExecutor,
                vwPredictorPool);
    }

    /**
     * Create a multilclass multiline predictor that distributes the work of
     * multiclass multiline prediction to backing thread pool specified as
     * executor service. One can use the predictor returned to predict with a
     * timeout.
     * 
     * A fixed size of poolSize thread pool will be created.
     * 
     * @param predictorName
     *            name of the predictor
     * @param learnerExecutor
     *            thread pool to be used
     * @param poolSize
     *            for predictor pool created internally.
     * @param command
     *            command arguments as you would have used in command line for
     *            initializing VW
     * @return VWConcurrentMultilabelsPredictor instance.
     * @throws InterruptedException
     */
    public static VWConcurrentMulticlassMultilinePredictor createMulticlassMultilinePredictorBase(
            final String predictorName,
            final int poolSize,
            final String command,
            final boolean predictNamedLabels) throws InterruptedException {
        return createMulticlassMultilinePredictorBase(predictorName, Executors.newFixedThreadPool(poolSize), poolSize,
                command, predictNamedLabels);
    }

    /**
     * Create a multilclass multiline predictor that distributes the work of
     * multiclass multiline prediction to backing thread pool specified as
     * executor service. One can use the predictor returned to predict with a
     * timeout.
     * 
     * This method offers flexibility to reuse an existing thread pool for
     * executing the predictors. In an ideal case, the total number of
     * predictors should be no less than the number of threads to achieve full
     * utilization. Otherwise memories will be wasted for holding predictors
     * that are not utilized.
     * 
     * @param predictorName
     *            name of the predictor
     * @param learnerExecutor
     *            thread pool to be used
     * @param poolSize
     *            for predictor pool created internally.
     * @param command
     *            command arguments as you would have used in command line for
     *            initializing VW
     * @return VWConcurrentMultilabelsPredictor instance.
     * @throws InterruptedException
     */
    public static VWConcurrentMulticlassMultilinePredictor createMulticlassMultilinePredictorBase(
            final String predictorName,
            final ExecutorService learnerExecutor, final int poolSize,
            final String command,
            final boolean predictNamedLabels) throws InterruptedException {
        final LinkedBlockingQueue<VWMulticlassLearner> vwPredictorPool = createVWPredictorPool(
                learnerExecutor, poolSize, command);
        return new VWConcurrentMulticlassMultilinePredictor(predictorName, learnerExecutor, vwPredictorPool,
                predictNamedLabels);
    }

    /**
     * Create a predictor pool based on the poolSize. This method waits till all the predictors are
     * initialized and depending on the size of the predictor, this could be matter of few seconds.
     * 
     * @param learnerExecutor
     * @param poolSize
     * @param command
     * @throws InterruptedException
     */
    private static <P extends VWLearner> LinkedBlockingQueue<P> createVWPredictorPool(
            final ExecutorService learnerExecutor, final int poolSize,
            final String command) throws InterruptedException {
        final LinkedBlockingQueue<P> vwPredictorPool = new LinkedBlockingQueue<P>(poolSize);
        final P seedLearner = VWLearners.create(command);
        for (int i = 1; i < poolSize; i++) {
            vwPredictorPool.add(VWLearners.clone(seedLearner));
        }
        vwPredictorPool.add(seedLearner);
        return vwPredictorPool;
    }
}
