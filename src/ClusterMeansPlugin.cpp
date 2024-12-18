#include "ClusterMeansPlugin.h"

#include <ClusterData/Cluster.h>
#include <ClusterData/ClusterData.h>
#include <PointData/PointData.h>

#include <actions/DatasetPickerAction.h>
#include <CoreInterface.h>
#include <Dataset.h>
#include <DataType.h>
#include <LinkedData.h>

#include <ankerl/unordered_dense.h>

#include <cassert>
#include <cmath>        // sqrt
#include <cstdint>
#include <exception>
#include <map>
#include <utility>      // move
#include <vector>

#include <QDebug>
#include <QList>
#include <QString>
#include <QVector>

using HashUInt32 = ankerl::unordered_dense::hash<std::uint32_t>;
using DenseSet = ankerl::unordered_dense::set<std::uint32_t, HashUInt32>;
using HashMap = ankerl::unordered_dense::map<std::uint32_t, std::uint32_t, HashUInt32>;

Q_PLUGIN_METADATA(IID "studio.manivault.ClusterMeansPlugin")

SelectInputDataDialog::SelectInputDataDialog(QWidget* parentWidget, const mv::Datasets& parents) :
    QDialog(parentWidget),
    _parentsAction(this, "Dataset"),
    _loadAction(this, "Create Dataset"),
    _assignToDirectParentAction(this, "Assign to immediate parent data"),
    _groupAction(this, "Settings")
{
    setWindowTitle(tr("Compute means from..."));

    _parentsAction.setDatasets(parents);

    _assignToDirectParentAction.setToolTip("If toggled, the ouput data will have\nthe same number of points as the input,\notherwise the number of clusters.");

    _loadAction.setEnabled(false);

    _groupAction.addAction(&_parentsAction);
    _groupAction.addAction(&_assignToDirectParentAction);
    _groupAction.addAction(&_loadAction);

    auto layout = new QVBoxLayout();
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(_groupAction.createWidget(this));
    setLayout(layout);

    connect(&_parentsAction, &mv::gui::DatasetPickerAction::datasetPicked, this, [this]() {
        _loadAction.setEnabled(_parentsAction.getCurrentDataset().isValid());
        });

    connect(&_loadAction, &mv::gui::TriggerAction::triggered, this, [this]() {
        accept();
        });
}

ClusterMeansPlugin::ClusterMeansPlugin(const PluginFactory* factory) :
    TransformationPlugin(factory)
{
}

void ClusterMeansPlugin::transform()
{
    // Get input
    auto clusterData = getInputDataset<Clusters>();

    // check if data set covers base data completely and fully
    DenseSet unique_elements;
    size_t maxDataID = 0;

    auto getGetUniqueAndDuplicates = [&maxDataID](const mv::Dataset<Clusters>& clusterData, DenseSet& unique_elements) -> bool
        {
            for (const auto& cluster : clusterData->getClusters()) {
                for (const auto id : cluster.getIndices())
                {
                    if(id > maxDataID)
                        maxDataID = id;

                    // Insert failed, indicating a duplicate
                    if (!unique_elements.insert(id).second)
                        return true;
                }
            }
            return false;
        };

    if (getGetUniqueAndDuplicates(clusterData, unique_elements))
    {
        qDebug() << "Contains duplicate IDs, cannot create means data";
        return;
    }

    size_t numClusterIDs = unique_elements.size();

    mv::Datasets possibleParents;
    for (auto parentItem : clusterData->getDataHierarchyItem().getAncestors())
    {
        if (parentItem == nullptr)
            continue;

        auto parentDataset = parentItem->getDataset();

        // we want points and full data
        if (parentDataset.getDataset()->getDataType() != PointType || !parentDataset.getDataset()->isFull())
            continue;

        const auto parentPoints = mv::Dataset<Points>(parentDataset);
        if (!parentPoints.isValid())
            continue;

        auto numDataPoints = parentPoints->getNumPoints();
        if (numDataPoints != numClusterIDs && numDataPoints < maxDataID)
            continue;

        possibleParents << parentDataset;
    }

    // datasets with the same selection group index as one of the parents might also be interesting
    DenseSet selectionGroupIDs;
    for (const auto& possibleParent : possibleParents)
        if (possibleParent->getGroupIndex() >= 0)
            selectionGroupIDs.insert(possibleParent->getGroupIndex());

    if (selectionGroupIDs.size() > 0) {
        for (const auto& selectionGroupID : selectionGroupIDs) {
            for (const auto& dataset : mv::data().getAllDatasets()) {
                if (dataset->getGroupIndex() == selectionGroupID && dataset.getDataset()->getDataType() == PointType) {
                    const auto datasetPoints = mv::Dataset<Points>(dataset);
                    if (datasetPoints.isValid() && datasetPoints->getNumPoints() != numClusterIDs)
                        continue;

                    if (possibleParents.contains(dataset))
                        continue;

                    possibleParents << dataset;
                }
            }
        }
    }

    // Ask user for dataset
    SelectInputDataDialog inputDialog(nullptr, possibleParents);
    inputDialog.setModal(true);

    // open dialog and wait for user input
    int ok = inputDialog.exec();

    if (ok == QDialog::Accepted ) {

        auto parentDataset = inputDialog.getParentData();

        if (!parentDataset.isValid())
        {
            qDebug() << "ClusterMeans: Please select a valid data set";
            return;
        }

        // Compute means and standard deviations
        auto parentPointsDataset    = mv::Dataset<Points>(parentDataset);
        const auto numDims          = parentPointsDataset->getNumDimensions();
        const auto numPointsParent  = parentPointsDataset->getNumPoints();
        QVector<Cluster>& clusters  = clusterData->getClusters();
        const auto numClusters      = clusters.size();

        qDebug() << "ClusterMeans: using " << parentDataset->getGuiName() << " with " << numDims << " dimension";

        for (Cluster& cluster : clusters)
        {
            const auto& indices         = cluster.getIndices();
            std::vector<float>& average = cluster.getMean();
            std::vector<float>& stddev  = cluster.getStandardDeviation();

            average.resize(numDims);
            stddev.resize(numDims);

            parentPointsDataset->visitData([this, numDims, &indices, &average, &stddev](auto pointData) {
                // averages
                for (const auto& i: indices) {
                    const auto& values = pointData[i];
                    for (int64_t dim = 0; dim < numDims; ++dim) {
                        average[dim] += values[dim];
                    }
                }

                for (auto& averageDim : average)
                    averageDim /= static_cast<float>(indices.size());

                // standard deviation
                for (const auto& i : indices) {
                    const auto& values = pointData[i];

                    for (int64_t dim = 0; dim < numDims; ++dim) {
                        const auto centered = values[dim] - average[dim];
                        stddev[dim] += (centered * centered);
                    }
                }

                for (auto& stddevDim : stddev)
                    stddevDim = std::sqrt(stddevDim / static_cast<float>(indices.size()));

                });

        }

        // Create new output
        Dataset<Points> meansData;
        const auto meansDataName = parentDataset->getGuiName() + " Cluster Means";

        std::vector<float> means;

        auto assignToDirectParent = [&meansData, &meansDataName, &numDims, &parentDataset, &numPointsParent, &clusters, &clusterData](std::vector<float>& means) {
            Dataset<Points> directParent = clusterData->getParent<Points>();
            const auto numPointsDirectParent = directParent->getNumPoints();

            meansData = mv::data().createDerivedDataset<Points>(meansDataName, directParent);
            means.resize(static_cast<size_t>(numDims)* numPointsDirectParent);

            const std::vector<mv::LinkedData>& linkedData = directParent->getLinkedData();
            bool useLinkedData = !linkedData.empty() && linkedData[0].getTargetDataset() == parentDataset;

            if (useLinkedData)
            {
                const std::map<std::uint32_t, std::vector<std::uint32_t>>& linkedMap = linkedData[0].getMapping().getMap();

                HashMap reverseLinkedData;
                reverseLinkedData.reserve(numPointsParent);

                std::uint32_t localID = 0;
                for (const auto& [globalID, mappedGlobalIDs] : linkedMap)
                {
                    reverseLinkedData.insert({ globalID , { localID } });
                    localID++;
                }

                for (const auto& cluster : clusters)
                {
                    const std::vector<float>& avgs = cluster.getMean();
                    const std::vector<std::uint32_t>& idx = cluster.getIndices();
                    for (const std::uint32_t globalID : idx)
                    {
                        if (!reverseLinkedData.contains(globalID))
                            continue;

                        std::copy(avgs.begin(), avgs.end(), means.data() + reverseLinkedData[globalID] * numDims);
                    }
                }

            }
            else
            {
                for (const auto& cluster : clusters)
                {
                    const auto& avgs = cluster.getMean();
                    for (const auto& id : cluster.getIndices())
                        std::copy(avgs.begin(), avgs.end(), means.data() + id * numDims);
                }
            }

            };

        auto assignToSelectedParent = [&meansData, &meansDataName, &numDims, &parentDataset, &numPointsParent, &clusters, &numClusters](std::vector<float>& means) {
            meansData = mv::data().createDataset<Points>("Points", meansDataName, parentDataset);

            // Get means from clusters
            for (const auto& cluster : clusters)
            {
                const auto& avgs = cluster.getMean();
                means.insert(means.end(), avgs.begin(), avgs.end());
            }

            assert(means.size() == numDims * numClusters);

            // Add selection maps
            mv::SelectionMap selectionMapMeansToParents;
            auto& mapMeansToParents = selectionMapMeansToParents.getMap();

            for (size_t clusterID = 0; clusterID < numClusters; clusterID++)
                mapMeansToParents[clusterID] = clusters[clusterID].getIndices();

            meansData->addLinkedData(parentDataset, selectionMapMeansToParents);

            };

        try
        {
            qDebug() << "ClusterMeans: creating data set " + meansDataName;

            if (inputDialog.assignToDirectParent())
                assignToDirectParent(means);
            else
                assignToSelectedParent(means);

            // Publish data
            meansData->setData(std::move(means), numDims);
            meansData->setDimensionNames(parentPointsDataset->getDimensionNames());
            events().notifyDatasetDataChanged(meansData);

        }
        catch (const std::exception& e)
        {
            qWarning() << "ClusterMeans: Failed. Caught exception: " << e.what();
        }

    }

}


// =============================================================================
// Plugin Factory 
// =============================================================================

TransformationPlugin* ClusterMeansPluginFactory::produce()
{
    return new ClusterMeansPlugin(this);
}

mv::DataTypes ClusterMeansPluginFactory::supportedDataTypes() const
{
    return { ClusterType };
}

mv::gui::PluginTriggerActions ClusterMeansPluginFactory::getPluginTriggerActions(const mv::Datasets& datasets) const
{
    mv::gui::PluginTriggerActions pluginTriggerActions;

    const auto numberOfDatasets = datasets.count();

    if (PluginFactory::areAllDatasetsOfTheSameType(datasets, ClusterType)) {
        if (numberOfDatasets >= 1 && datasets.first()->getDataType() == ClusterType) {

            auto pluginTriggerAction = new mv::gui::PluginTriggerAction(const_cast<ClusterMeansPluginFactory*>(this), this, QString("Create Mean Dataset"), QString("Create Mean Dataset"), getIcon(), [this, datasets](mv::gui::PluginTriggerAction& pluginTriggerAction) -> void {
                for (const auto& dataset : datasets) {
                    auto pluginInstance = dynamic_cast<ClusterMeansPlugin*>(plugins().requestPlugin(getKind()));

                    pluginInstance->setInputDataset(dataset);
                    pluginInstance->transform();
                }
                });

            pluginTriggerActions << pluginTriggerAction;

        }
    }

    return pluginTriggerActions;
}