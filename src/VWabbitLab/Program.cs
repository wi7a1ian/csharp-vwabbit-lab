
using System.Diagnostics;
using VW;
using VW.Labels;
using VW.Serializer.Attributes;

/*
https://github.com/VowpalWabbit/vowpal_wabbit/wiki/C%23-Binding

VW is using L-BGFS optimization method that stand for “Limited-memory Broyden–Fletcher–Goldfarb–Shanno”.
BGFS uses more memory than L-BGFS and even more than Conjugate Gradient. Both are more complex to implement than Stochastic Gradient Descent. Stochastic Gradient Descent is Gradient Descent with momentum.

TODO: take large dataset and use text tokenization split words --> N * Feature(hash(word), 1.0f + (float) Math.Log10(count))

*/

internal class Program
{
    // Simple classification example
    private static void Main(string[] args)
    {
        var dataset = new[]{
            new Document { Author = "Broyden", Text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Fringilla urna porttitor rhoncus dolor purus non. 1999", Year = 1999 },
            new Document { Author = "Fletcher", Text = "Tristique magna sit amet purus gravida quis blandit. 1989", Year = 1989 },
            new Document { Author = "Goldfarb", Text = "Senectus et netus et malesuada fames ac turpis. Nunc id cursus metus aliquam eleifend mi in nulla. Interdum consectetur libero id faucibus nisl tincidunt eget. Egestas diam in arcu cursus euismod quis viverra nibh cras. 1960", Year = 1960 }
        };
        
        var similarToFirst = new Document { Author = "Shanno", Text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.", Year = 2019 };
        
        var classification = new[]{
            0.0f,
            1.0f,
            1.0f
        };

        // Using defined data type
        using (var session = new VowpalWabbit<Document>("-f test1.model"))
        {
            for (int epochNr = 0; epochNr < 100; epochNr++)
            for (int i = 0; i < dataset.Length; ++i)
            {
                session.Learn(dataset[i], new SimpleLabel { Label = classification[i] });
            }

            var prediction = session.Predict(dataset[0], VowpalWabbitPredictionType.Scalar);
            Debug.Assert(Math.Round(prediction, 1).Equals(classification[0]));
        }

        // Direct approach using exmaple builder
        using (var session = new VowpalWabbit("-f test2.model"))
        {
            for (int epochNr = 0; epochNr < 100; epochNr++)
            for (int i = 0; i < dataset.Length; ++i)
            {
                using var exampleBuilder = new VowpalWabbitExampleBuilder(session);
                SetupExample(session, exampleBuilder, dataset[i], classification[i]);
                using var example = exampleBuilder.CreateExample();
                session.Learn(example);
            }

            // exact value
            using (var exampleBuilder = new VowpalWabbitExampleBuilder(session))
            {
                SetupExample(session, exampleBuilder, dataset[0]);
                using var example = exampleBuilder.CreateExample();
                var prediction = session.Predict(example, VowpalWabbitPredictionType.Scalar);
                Debug.Assert(Math.Round(prediction, 1).Equals(classification[0])); // should be classified as the first one
            }

            // similar value
            using (var exampleBuilder = new VowpalWabbitExampleBuilder(session))
            {
                SetupExample(session, exampleBuilder, similarToFirst);
                using var example = exampleBuilder.CreateExample();
                var prediction = session.Predict(example, VowpalWabbitPredictionType.Scalar);
                Debug.Assert(Math.Round(prediction).Equals(classification[0])); // should be classified almost as the first one
            }
        }
    }

    private static void SetupExample(VowpalWabbit session, VowpalWabbitExampleBuilder exampleBuilder, Document document, float? classifiation = null)
    {
        using (var ns = exampleBuilder.AddNamespace('n'))
        {
            var namespaceHash = session.HashSpace("ns0");
            ns.AddFeature(session.HashFeature(nameof(Document.Author), namespaceHash), document.Author.GetHashCode());

            // ns.AddFeature(session.HashFeature(nameof(Document.Text), namespaceHash), document.Text.GetHashCode());
            foreach(var (feature, value) in ExtractFeaturesFromText(document.Text))
            {
                ns.AddFeature(session.HashFeature(feature, namespaceHash), value);
            }
            ns.AddFeature(session.HashFeature(nameof(Document.Year), namespaceHash), document.Year.GetHashCode());
        }

        if(classifiation is not null)
        {
            exampleBuilder.ApplyLabel(new SimpleLabel { Label = classifiation.Value });
        }
    }

    private static IDictionary<string, int> ExtractFeaturesFromText(string text)
    {
        var tokens = Tokenize(text);
        return tokens.Distinct().ToDictionary( k => k.ToString(), v => tokens.Count( t => t.Equals(v)));
    }

    private static List<int> Tokenize(string text)
        => text.ToUpper().Split(" ").Where( t => !string.IsNullOrWhiteSpace(t)).Select( t => t.GetHashCode()).ToList();
}

public class Document
{
    [Feature(FeatureGroup = 'n', Namespace = "ns0")]
    public string Author { get; set; }

    // separate namespace just in case author or year also appears in the text
    [Feature(FeatureGroup = 'n', Namespace = "ns1")]
    public string Text { get; set; }

    [Feature(FeatureGroup = 'n', Namespace = "ns0")]
    public int Year { get; set; }
}