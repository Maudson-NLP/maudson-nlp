$(function(){
   // Bind advanced options click listener
    $('#show-advanced').on('click', function(){
       $('#advanced-options').toggle();
    });
});


function doUpload(){
    $('#summarization-result').text('Summarizing...');

    var fileInput = document.getElementById('the-file');
    var file = fileInput.files[0];
    var formData = new FormData();

    colnames = $('#the-column-names').val();
    formData.append('file', file);
    formData.append('columns', colnames);
    formData.append('l-value', $('#l-value').val());
    formData.append('ngram-min', $("#ngram-min").val());
    formData.append('ngram-max', $("#ngram-max").val());
    formData.append('tfidf', $("#tfidf").prop("checked"));
    formData.append('use-svd', $("#svd").prop("checked"));
    formData.append('scale-vectors', $("#scale-vectors").prop("checked"));
    formData.append('top-k', $("#top-k").val());
    formData.append('use-noun-phrases', $("#noun-phrases").prop("checked"));
    formData.append('split-longer-sentences', $("#split-longer-sentences").prop("checked"));
    formData.append('to-split-length', $("#to-split-length").val());
    formData.append('group-by', $("#group-by").val());
    formData.append('extract-sibling-sents', $("#extract-sibling-sents").prop("checked"));

    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {

            if (xhr.status === 200) {
                var json = JSON.parse(xhr.responseText);
                // Join the array of strings by simple spaces - todo
                json.map(function(j){ j[1] = j[1].join(' '); return j });

                var $compiled = $('#summarization--template').tmpl({
                    summaries: json
                });
                $('#summarization-result').empty().append($compiled);
            } else {
                $('#summarization-result').empty().append(xhr.statusText);
                console.log("Error", xhr.statusText);
            }

        }
    };

    // Add any event handlers here...
    xhr.open('POST', '/summarize', true);
    xhr.send(formData);
}