#pragma once

#include <GL/glew.h>
#include <freetype/freetype.h>

#include <string>
#include <vector>
#include <ranges>
#include <algorithm>
#include <numeric>

#include "Geometry.h"

class Text {
public:

	std::size_t mX = 0, mY = 0;

	FT_Face mFontFace;

	FT_Int mTotalPixelsWidth = 0, mTotalPixelsHeight = 0;
	std::vector<std::uint32_t> mTextPixelsBuffer;

	Text(std::size_t x, std::size_t y, const FT_Face& fontFace)
		: mX(x), mY(y), mFontFace(fontFace)
	{}

	bool setText( const std::string& text) {

		if (text.empty()) return false;

		FT_Int totalPixelsWidth = 0
			, maxAscent = 0
			, maxDescent = 0
			, totalPixelsHeight = 0;

		auto getTextDimensions = [&]() {
			FT_GlyphSlot glyph;
			for (auto c : text) {

				if (std::isspace(c))
					FT_Load_Char(mFontFace, '_', FT_LOAD_RENDER);
				else
					FT_Load_Char(mFontFace, c, FT_LOAD_RENDER);

				glyph = mFontFace->glyph;
				maxAscent = std::max(maxAscent, glyph->bitmap_top);
				maxDescent = std::max(maxDescent, FT_Int(glyph->bitmap.rows - glyph->bitmap_top));
				totalPixelsWidth += glyph->advance.x >> 6;
			}
			totalPixelsHeight = maxAscent + maxDescent;
			};
		
		getTextDimensions();

		std::uint32_t whiteColor = 0xFFFFFFFF;//opaque with white background with grey forground/text
		std::uint8_t* whiteBytes = reinterpret_cast<std::uint8_t*>(&whiteColor);

		//add spaces between glyphs and padding of line and bottom of line
		totalPixelsWidth += text.size()*2 +2;
		totalPixelsHeight += 2;

		bool resized =   totalPixelsWidth > mTotalPixelsWidth || totalPixelsHeight > mTotalPixelsHeight;

		mTotalPixelsWidth = std::max(mTotalPixelsWidth, totalPixelsWidth);
		mTotalPixelsHeight = std::max(mTotalPixelsHeight, totalPixelsHeight);
		
		mTextPixelsBuffer.clear();
		mTextPixelsBuffer.resize(mTotalPixelsWidth * mTotalPixelsHeight, whiteColor);

		auto drawText = [&]() {

			auto darkenPixel = [&](auto savedColor, float darkenScale) {
				savedColor[0] = whiteBytes[0] * darkenScale; // R
				savedColor[1] = whiteBytes[1] * darkenScale;   // G
				savedColor[2] = whiteBytes[2] * darkenScale; // B
				savedColor[3] = whiteBytes[3]; // A
				};

			int penX = 0;
			FT_GlyphSlot glyph;

			for (std::size_t i : std::views::iota(0ULL, text.size())) {

				char c = text[i];

				if (std::isspace(c)) {
					FT_Load_Char(mFontFace, '_', FT_LOAD_RENDER);
					glyph = mFontFace->glyph;
				} else {

					FT_Load_Char(mFontFace, c, FT_LOAD_RENDER);
					glyph = mFontFace->glyph;
					FT_Bitmap& bmp = glyph->bitmap;

					int xOffset = penX + glyph->bitmap_left;
					int yOffset = maxAscent - glyph->bitmap_top;

					for (int row = 0; row < bmp.rows; ++row) 
						for (int col = 0; col < bmp.width; ++col) {
							int x = 1 + xOffset + col + i * 2; //+i/1 == space between glyphs
							int y = 1 + yOffset + row;

							float darkenScale = 1.0f - bmp.buffer[row * bmp.pitch + col] / 255.0f;
							std::uint8_t* savedColor = reinterpret_cast<std::uint8_t*>(&mTextPixelsBuffer[y * mTotalPixelsWidth + x]);

							darkenPixel(savedColor, darkenScale);
						}
				}

				penX += glyph->advance.x >> 6;
			}

			};

		drawText();

		//return if size increased
		return resized;
	} 
	void destroy() {
		mTextPixelsBuffer.clear();
		mTextPixelsBuffer.shrink_to_fit();
	}
	float getAspectRatio() {
		return mTotalPixelsWidth / float(mTotalPixelsHeight);
	}
};

class TextArea {
public:
	std::vector<Text> mLabels;
	using LabelReference = std::size_t;

	using LabeledValue = std::pair<Text, Text>;
	std::vector<LabeledValue> mLabeledValues;
	using LabeledValueReference = std::size_t;

	QuadManager::QuadReference mQuadReference;

	GLuint mTexture = 0;
	FT_Int mTotalPixelsWidth = 0, mTotalPixelsHeight = 0;

	std::size_t mInsertY = 0; //start at top

	float mScale = 0.01f;
	bool mUpdateLabels = true;

	void create(QuadManager& qm, float scale = 1.0) {
		mScale = scale;

		calculateDimensions();

		auto YLineNum = mLabeledValues.size() + mLabels.size();
		mQuadReference = qm.add(-1.0, -1.0, mTotalPixelsWidth / float(mTotalPixelsHeight), scale * YLineNum);
	}

	void createTexture(std::size_t width, std::size_t height) {

		if (mTexture)
			glDeleteTextures(1, &mTexture);

		glGenTextures(1, &mTexture);

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, mTexture);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mTotalPixelsWidth, mTotalPixelsHeight, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, nullptr);

		GLubyte color[4] = { 255,255,255,255 };
		glClearTexImage(mTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, color);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glBindTexture(GL_TEXTURE_2D, 0);
	}
	void destroy() {

		for (auto& label : mLabels)
			label.destroy();

		for (auto& [caption, value] : mLabeledValues) {

			caption.destroy();
			value.destroy();
		}

		if (mTexture) {
			glDeleteTextures(1, &mTexture);
			mTexture = 0;
		}
	}

	void unionDimensions(const Text& text) {
		mTotalPixelsWidth = std::max(mTotalPixelsWidth, FT_Int( text.mX + text.mTotalPixelsWidth));
		mTotalPixelsHeight = std::max(mTotalPixelsHeight, FT_Int( text.mY + text.mTotalPixelsHeight));
	}
	void calculateDimensions() {

		//loop through all text components and get the bounding rectangle including the old one, get bigger not smaller

		FT_Int oldWidth = mTotalPixelsWidth, oldHeight = mTotalPixelsHeight;

		for (const auto& label : mLabels)
			unionDimensions(label);

		for (const auto& [label, caption] : mLabeledValues) {
			unionDimensions(label);
			unionDimensions(caption);
		}

		if (mTotalPixelsWidth == oldWidth && mTotalPixelsHeight == oldHeight) return;

		createTexture(mTotalPixelsWidth, mTotalPixelsHeight);

		mUpdateLabels = true;
	}
	void render(QuadManager& qm, bool updateLabels = false) {

		glBindTexture(GL_TEXTURE_2D, mTexture);

		auto updateTexture = [&](auto& text) {
			glTexSubImage2D(GL_TEXTURE_2D, 0, text.mX, text.mY, text.mTotalPixelsWidth, text.mTotalPixelsHeight, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, text.mTextPixelsBuffer.data());
			};
		
		updateLabels |= mUpdateLabels;

		if (updateLabels) {
			for (const auto& label : mLabels)
				updateTexture(label);

			for (const auto& [label, value] : mLabeledValues)
				updateTexture(label);

			mUpdateLabels = false;
		}
		for (const auto& [label, value] : mLabeledValues)
			updateTexture(value);

		qm.render(mQuadReference);
	} 

	LabelReference addLabel(const FT_Face& fontFace, const std::string& text) {
		
		Text label(0, mInsertY, fontFace);
		label.setText(text);

		mInsertY += label.mTotalPixelsHeight;

		mLabels.push_back(std::move(label));

		return mLabels.size() - 1;
	}

	LabeledValueReference addLabeledValue(const FT_Face& fontFace, const std::string& text, const std::string& valueText) {
		
		std::size_t insertX = 0;

		Text label(insertX, mInsertY, fontFace);
		label.setText(text);

		insertX = label.mTotalPixelsWidth;

		Text value(insertX, mInsertY, fontFace);
		value.setText(valueText);

		mLabeledValues.push_back({ std::move(label), std::move(value) });
		                 
		mInsertY += value.mTotalPixelsHeight;

		return mLabeledValues.size() - 1;
	}

	void updateLabeledValue(LabeledValueReference labeledValueRef, const std::string& valueText) {
		auto& value = mLabeledValues[labeledValueRef].second;
		auto resized = value.setText(valueText);
		if (resized) calculateDimensions();
		
	}
	void updateLabel(LabelReference labelRef, const std::string& labelText) {
		auto& label = mLabels[labelRef];
		auto resized = label.setText(labelText);
		if (resized) calculateDimensions();
	}
};
class TextManager {

public:

	static constexpr auto mMinecraftFontName = "./minecraft.ttf";

	FT_UInt mFontSize = 12;

	QuadManager* mQuadManager;

	FT_Library mFreeType;
	FT_Face mMinscraftFontFace;

	using TextAreaReference = std::size_t;
	std::vector<TextArea> mTextAreas;

	bool mVisible = true;

	void create(QuadManager* quadManager, float scale = 0.01f) {

		mQuadManager = quadManager;
		
		FT_Init_FreeType(&mFreeType);

		FT_New_Face(mFreeType, mMinecraftFontName, 0, &mMinscraftFontFace);
		FT_Set_Pixel_Sizes(mMinscraftFontFace, 0, mFontSize);
	}

	void destroy() {

		FT_Done_Face(mMinscraftFontFace);       // Destroys the font face
		FT_Done_FreeType(mFreeType);     // Shuts down the FreeType library
	}

	TextAreaReference addTextArea() {
		mTextAreas.push_back({});
		return mTextAreas.size()-1;
	}
	TextArea& getTextArea(TextAreaReference ref) {
		return mTextAreas[ref];
	}
	void render(bool updateLabels = false) {

		if (!mVisible) return;

		auto& qm = *mQuadManager;
		 
		//glEnable(GL_BLEND);
		//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		for (auto& txtArea : mTextAreas)
			txtArea.render(qm, updateLabels);

		glBindTexture(GL_TEXTURE_2D, 0);
	} 
	void show() {
		mVisible = true;
	}
	void hide() {
		mVisible = false;
	}
	void toggleVisibility() {
		mVisible = !mVisible;
	}
};