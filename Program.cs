// See https://aka.ms/new-console-template for more information

using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration.Attributes;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TagPredPair = (WaifuDiffusion.Tag Tag, float Pred);

namespace WaifuDiffusion;

public class Program : IDisposable
{
    public static bool GeneralMCutEnabled { get; } = false;

    public static bool CharacterMCutEnabled { get; } = false;

    public static string ProjectPath { get; } = @"C:\WorkSpace\WaifuDiffusion-Tagger\";

    public static string TagsCsv => ProjectPath + "selected_tags.csv";

    public static string ModelOnnx => ProjectPath + "model.onnx";

    public IReadOnlyList<Tag>? TagsList { get; set; }

    public InferenceSession? Model { get; set; }

    public int ImageTargetSize { get; set; } = -1;

    public string? InputName { get; set; } 

    public static async Task Main()
    {
        using var program = new Program();
        await program.PrepareModelAsync();
        var imagePath = @"C:\Users\poker\Pictures\[ZS] 127096707.png";
        var (rating, characterRes, generalRes) = await program.RunAsync(imagePath);

        var format = string.Join(", ", generalRes.OrderByDescending(x => x.Pred).Select(t => t.Tag.Name));
        Console.WriteLine($"General Tags: {format}");

        foreach (var (key, value) in rating)
            Console.WriteLine($"{key.Name}: {value}");
        foreach (var (key, value) in characterRes)
            Console.WriteLine($"{key.Name}: {value}");
        foreach (var (key, value) in generalRes)
            Console.WriteLine($"{key.Name}: {value}");
    }

    [MemberNotNull(nameof(TagsList), nameof(Model), nameof(InputName))]
    public async Task PrepareModelAsync()
    {
        using var streamReader = new StreamReader(TagsCsv);
        using var csvReader = new CsvReader(streamReader, CultureInfo.CurrentCulture);
        var tagsList = new List<Tag>();
        while (await csvReader.ReadAsync())
        {
            var t = csvReader.GetRecord<Tag>();
            if (!Kaomojis.Contains(t.Name))
                t.Name = t.Name.Replace('_', ' ');
            tagsList.Add(t);
        }
        TagsList = tagsList;

        Model = new(ModelOnnx);

        InputName = Model.InputNames[0];
        if (Model.InputMetadata[InputName].Dimensions is [_, var height, var width, _])
            ImageTargetSize = height;
    }

    public async Task<(TagPredPair[] Rating, TagPredPair[] CharacterRes, TagPredPair[] GeneralRes)> RunAsync(string imagePath)
    {
        using var image = await Image.LoadAsync<Rgba32>(imagePath);
        var preparedImage = PrepareImage(image);

        // Run the modelv
        var inputTensor = preparedImage.ToTensor();
        using var results = Model.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(InputName, inputTensor) });
        var preds = results[0].AsEnumerable<float>().ToArray();

        // Map predictions to labels
        var labels = TagsList.Zip(preds, (name, pred) => (Tag: name, Pred: pred)).ToArray();

        var lookup = labels.ToLookup(t => t.Tag.Category);

        // Process ratings
        var rating = lookup[Category.Rating].ToArray();

        // Process general tags
        var generalNames = lookup[Category.General].ToArray();
        var generalThresh = 0.5f;
        if (GeneralMCutEnabled)
        {
            var generalProbs = generalNames.Select(x => x.Pred).ToArray();
            generalThresh = MCutThreshold(generalProbs);
        }
        var generalRes = generalNames.Where(x => x.Pred > generalThresh).ToArray();

        // Process character tags
        var characterNames = lookup[Category.Character].ToArray();
        var characterThresh = 0.5f;

        if (CharacterMCutEnabled)
        {
            var characterProbs = characterNames.Select(x => x.Pred).ToArray();
            characterThresh = Math.Max(0.15f, MCutThreshold(characterProbs));
        }
        var characterRes = characterNames.Where(x => x.Pred > characterThresh).ToArray();

        return (rating, characterRes, generalRes);
    }

    public float[,,,] PrepareImage(Image<Rgba32> image)
    {
        // Convert RGBA to RGB
        var rgbImage = image.Clone(ctx => ctx.BackgroundColor(Color.White));

        // Pad image to square
        var maxDim = Math.Max(rgbImage.Width, rgbImage.Height);
        var padLeft = (maxDim - rgbImage.Width) / 2;
        var padTop = (maxDim - rgbImage.Height) / 2;

        var paddedImage = new Image<Rgb24>(maxDim, maxDim, new Rgb24(0xff, 0xff, 0xff));
        paddedImage.Mutate(ctx => ctx.DrawImage(rgbImage, new Point(padLeft, padTop), 1f));

        // Resize
        if (maxDim != ImageTargetSize) 
            paddedImage.Mutate(ctx => ctx.Resize(ImageTargetSize, ImageTargetSize, KnownResamplers.Bicubic));

        // 转换为浮点数组 (RGB 转 BGR)
        var imageArray = new float[1, ImageTargetSize, ImageTargetSize, 3];
        paddedImage.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (var x = 0; x < accessor.Width; x++)
                {
                    var pixel = row[x];
                    imageArray[0, y, x, 0] = pixel.B; // B
                    imageArray[0, y, x, 1] = pixel.G; // G
                    imageArray[0, y, x, 2] = pixel.R; // R
                }
            }
        });

        return imageArray;
    }

    /// <summary>
    /// Maximum Cut Thresholding (mCut)
    /// Reference: Largeron, C., Moulin, C., & Gery, M. (2012). mCut: A Thresholding Strategy
    /// for Multi-label Classification. In 11th International Symposium, IDA 2012 (pp. 172-183).
    /// </summary>
    /// <param name="probs">Array of probabilities.</param>
    /// <returns>Calculated threshold.</returns>
    public static float MCutThreshold(float[] probs)
    {
        // Sort probabilities in descending order
        var sortedProbs = probs.OrderByDescending(p => p).ToArray();

        // Calculate differences between consecutive probabilities
        var diffs = new float[sortedProbs.Length - 1];
        for (var i = 0; i < sortedProbs.Length - 1; i++)
        {
            diffs[i] = sortedProbs[i] - sortedProbs[i + 1];
        }

        // Find the index of the maximum difference
        var t = Array.IndexOf(diffs, diffs.Max());

        // Calculate the threshold as the average of the two probabilities at the cut
        var threshold = (sortedProbs[t] + sortedProbs[t + 1]) / 2;

        return threshold;
    }

    public static HashSet<string> Kaomojis { get; } =
    [
        "0_0",
        "(o)_(o)",
        "+_+",
        "+_-",
        "._.",
        "<o>_<o>",
        "<|>_<|>",
        "=_=",
        ">_<",
        "3_3",
        "6_9",
        ">_o",
        "@_@",
        "^_^",
        "o_o",
        "u_u",
        "x_x",
        "|_|",
        "||_||",
    ];

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        Model.Dispose();
    }
}

public record struct Tag
{
    [Name("tag_id")]
    public int TagId { get; set; }

    [Name("name")]
    public string Name { get; set; }

    [Name("category")]
    public Category Category { get; set; }

    [Name("count")]
    public int Count { get; set; }
}

public enum Category
{
    Rating = 9,
    General = 0,
    Character = 4
}
